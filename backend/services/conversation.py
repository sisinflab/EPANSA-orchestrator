import os
import logging
from typing import TypedDict, Annotated, Sequence, OrderedDict
import re

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from backend.services.analyzer import img_analyzer
from backend.services import chat_history_service as hist_service
from backend.core.config import settings
from backend.models.payloads import ChatResponse
from backend.services.recommender.recommender_agent import RecommenderAgent
from backend.services.graphRag_chat import chatbot
from backend.services.constants import (AGENT_LLM_CONFIG,
                                        CHATBOT_LLM_CONFIG,
                                        CHATBOT_EMBEDDING_CONFIG,
                                        RECOMMENDER_LLM_CONFIG,
                                        TMP_DIR)
from graphrag.query.context_builder.conversation_history import ConversationHistory

logger = logging.getLogger(__name__)


class FixedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            oldest = next(iter(self))
            del self[oldest]

    def delete_item(self, key):
        if key in self:
            del self[key]
            return True
        return False


# Caching for agents
agent_cache = FixedSizeDict(max_size=10)


def get_llm(model_env_value):
    if model_env_value.startswith('groq'):
        provider, model_name, base_url, api_key = model_env_value.split(",")
        llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=0,
        )

    elif model_env_value.startswith('gemini'):
        provider, model_name, _, api_key = model_env_value.split(",")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        llm = llm.bind(
            generation_config={
                "thinking_config": {
                    "thinkingBudget": 20000,
                }
            }
        )

    elif model_env_value.startswith('ollama'):
        provider, model_name, base_url, _ = model_env_value.split(",")
        llm = ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=0
        )

    elif model_env_value.startswith('openai'):
        provider, model_name, api_key, _ = model_env_value.split(",")
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0,
        )

    elif model_env_value.startswith('anthropic'):
        provider, model_name, api_key, _ = model_env_value.split(",")
        llm = ChatAnthropic(
            api_key=api_key,
            model=model_name,
            temperature=0,
            timeout=None
        )

    else:
        provider, model_name, api_endpoint, api_key = model_env_value.split(",")
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_endpoint,
            model=model_name,
            temperature=0
        )

    return llm


# --------- Lang-Graph Agent --------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_evaluation: str | None  # 'good' or 'bad'

class WorkflowConfigs:
    def __init__(self, recommender_agent: RecommenderAgent, user_id: int):
        self.recommender_agent = recommender_agent
        self.database = recommender_agent.database
        self.embedding_model = recommender_agent.embedding_model
        self.user_id = user_id


        @tool
        async def img_deep_analysis(question: str) -> str:
            """
            Useful when the user requests images analysis.
            Takes in input the user question and, first searches the best images to analyze between those stored in DB
            (matching the user query), then analyze them with a Visual Analyzer AI model and returns a response.
            """
            try:
                logging.info("Starting deep image analysis...")
                history = hist_service.get_history(self.user_id)
                photo_list_str = await chatbot(
                    query=question,
                    conversation_history=ConversationHistory.from_list(history[:-1]),
                    mode="img_deep_analysis",
                    user_input_dir=f"{TMP_DIR}/user-{self.user_id}/gRag/output",
                    model_env_value=CHATBOT_LLM_CONFIG,
                    embedding_env_value=CHATBOT_EMBEDDING_CONFIG,
                )

                # takes in input the bullet point text containing photos by GraphRAG and returns the list of photo_ids
                photo_ids = [line.strip()[2:] for line in photo_list_str.strip().splitlines() if line.strip().startswith('- ')]
                # analyzes the top similar photos (given the query) one by one and returns possible responses
                final_response = await img_analyzer(question=question, photo_ids=photo_ids, user_id=self.user_id)

                logging.info("Images analysis completed.")
                return final_response
            except Exception as e:
                logger.info(f"Error in img_deep_analysis for user {self.user_id}: {e}", exc_info=True)
                return "I'm sorry, I encountered an error while trying to analyze the images. Please try again later."

        @tool
        async def graph_RAG(question: str) -> str:
            """
            Takes in input the user question and returns a response based on his personal information
            (documents, photos, calendar events, alarms, etc.). Useful for personal but also generic questions.
            """
            try:
                logging.info("Starting RAG on graph process...")
                history = hist_service.get_history(self.user_id)
                res = await chatbot(
                    query=question,
                    conversation_history=ConversationHistory.from_list(history[:-1]),
                    mode="rag",
                    user_input_dir=f"{TMP_DIR}/user-{self.user_id}/gRag/output",
                    model_env_value=CHATBOT_LLM_CONFIG,
                    embedding_env_value=CHATBOT_EMBEDDING_CONFIG,
                )
                logging.info("RAG on graph completed.")
                return res
            except Exception as e:
                logger.info(f"Error in graph_RAG for user {self.user_id}: {e}", exc_info=True)
                return "I'm sorry, I encountered an error while trying to retrieve the information. Please try again later."

        @tool
        async def PoI_recommender(question: str) -> str:
            """
            Useful when the user requests recommendations or advices about trips and places to visit.
            Takes in input the user question and returns a list of suggested Points-of-Interest and places to visit
            that better fits his personality.
            """
            logger.info(f"PoI_recommender called for user_id: {self.user_id} with question: '{question}'")
            try:
                # use the already initialized recommender_agent
                response = await self.recommender_agent.get_response(question)
                logger.info(f"PoI_recommender generated response: {response}")
                return response
            except Exception as e:
                logger.info(f"Error in PoI_recommender for user {self.user_id}: {e}", exc_info=True)
                return "I'm sorry, I encountered an error while trying to find recommendations for you. Please try again later."

        self.primary_tools = [graph_RAG, PoI_recommender]
        self.fallback_tool = img_deep_analysis


class ReactAgent:
    def __init__(self, workflow_configs: WorkflowConfigs):
        self.primary_tools = workflow_configs.primary_tools
        self.fallback_tool = workflow_configs.fallback_tool
        self.llm = get_llm(AGENT_LLM_CONFIG)

        self.llm_w_primary_tools = self.llm.bind_tools(self.primary_tools)
        self.llm_w_fallback_tool = self.llm.bind_tools([self.fallback_tool])

    # GRAPH NODES
    async def standard_agent(self, state: AgentState) -> AgentState:
        """Main agent: it chooses between RAG and PoI_recommender"""
        response = await self.llm_w_primary_tools.ainvoke(state["messages"])
        return {"messages": [response], "rag_evaluation": state["rag_evaluation"]}

    async def fallback_agent(self, state: AgentState) -> AgentState:
        """LLM fallback node: called if Rag fails"""
        tmp_message = HumanMessage(
            content="""The previous search for information failed or yielded a vague answer. 
            You must now use the 'img_deep_analysis' tool to find a better answer to the user's original question."""
        )
        messages_with_context = state["messages"] + [tmp_message]
        response = await self.llm_w_fallback_tool.ainvoke(messages_with_context)
        return {"messages": [response], "rag_evaluation": state["rag_evaluation"]}

    async def evaluate_rag_response(self, state: AgentState) -> AgentState:
        """
        Response by GraphRAG evaluation.
        """
        user_question = state["messages"][0].content
        tool_output = state["messages"][-1].content

        evaluation_prompt = f"""
        Given the user's original question and the response provided by an AI agent, evaluate whether the agent managed to answer the question or not.

        User Question: "{user_question}"
        
        AI agent Response: "{tool_output}"
        
        If the AI agent says it cannot answer the question due to lack of information, or gives a vague answer, respond with the word “bad.”
        If it manages to fulfill the user’s request, respond with the word “good.”
        
        Here are some examples of unsatisfactory or vague answers:
        - “I don't have information about...”
        - "There's no mention about ..."
        - "There's no data about ..."
        - "I'm sorry but ..."
        - "It seems there was an issue retrieving the data ..."
        
        Reply with a single word only: “bad” if the response is unsatisfactory, or “good” if it is satisfactory.
        """

        evaluation_response = await self.llm.ainvoke(evaluation_prompt)
        decision = evaluation_response.content.strip().lower()
        decision = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", decision, flags=re.DOTALL | re.IGNORECASE)
        if decision not in ["bad", "good"]:
            raise Exception(f"Invalid response from evaluate_rag_response node: {decision}")

        return {"messages": [], "rag_evaluation": decision}

    # CONDITIONAL ROUTERS
    def router_after_llm(self, state: AgentState) -> str:
        """Decides whether to call a tool or terminate."""
        if state["messages"][-1].tool_calls:
            return "tools"
        return "end"

    def router_after_tools(self, state: AgentState) -> str:
        """
        Decides where to go after running a tool.
        If graph_RAG has been run, it goes to the evaluation node.
        Otherwise, it returns to the standard_agent.
        """
        tool_message = state["messages"][-1]
        if tool_message.name == 'graph_RAG':
            return "evaluate_rag"
        return "standard_agent"

    def router_after_evaluation(self, state: AgentState) -> str:
        """
        Decides whether to use fallback or terminate, based on the evaluation.
        """
        if state["rag_evaluation"] == "bad":
            return "fallback_agent"
        return "standard_agent"


    # GRAPH CREATION
    def create_agent(self, checkpointer):
        graph = StateGraph(AgentState)
        graph.add_node("standard_agent", self.standard_agent)
        all_tools = self.primary_tools + [self.fallback_tool]
        tools_node = ToolNode(tools=all_tools)
        graph.add_node("tools", tools_node)
        graph.add_node("evaluate_rag", self.evaluate_rag_response)
        graph.add_node("fallback_agent", self.fallback_agent)

        graph.add_edge(START, "standard_agent")
        graph.add_conditional_edges(
            "standard_agent",
            self.router_after_llm,
            {"tools": "tools", "end": END},
        )

        graph.add_conditional_edges(
            "tools",
            self.router_after_tools,
            {
                "evaluate_rag": "evaluate_rag",
                "standard_agent": "standard_agent"
            },
        )

        graph.add_conditional_edges(
            "evaluate_rag",
            self.router_after_evaluation,
            {
                "fallback_agent": "fallback_agent",
                "standard_agent": "standard_agent"
            },
        )

        graph.add_edge("fallback_agent", "tools")
        return graph.compile(checkpointer=checkpointer)



async def process_user_command(user_id: int, db_name: str, command_text: str, embedding_model) -> ChatResponse:
    """
    Processes a user's chat command, managing conversation history and agent caching.
    """
    try:
        hist_service.add_message(user_id=user_id, role="user", content=command_text)

        cache_key = user_id
        if cache_key in agent_cache:
            logger.info(f"Agent and profile cache hit for user_id: {user_id}")
            graph_agent, recomm_agent = agent_cache[cache_key]
        else:
            logger.info(f"New agent and profile creation for user_id: {user_id}")

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            poi_data_path = os.path.join(project_root, settings.RECOMMENDER_DATA_PATH)

            # create and initialize Recommender Agent
            recomm_agent = RecommenderAgent(
                database=db_name,
                user_id=user_id,
                embedding_model=embedding_model,
                model_env_value=RECOMMENDER_LLM_CONFIG,
                poi_data_path=poi_data_path,
            )
            await recomm_agent.initialize()

            # create LangGraph React Agent
            memory = MemorySaver()
            # pass the already initialized recommender_agent
            wf = WorkflowConfigs(recommender_agent=recomm_agent, user_id=user_id)
            ra = ReactAgent(wf)
            graph_agent = ra.create_agent(checkpointer=memory)

            # save in cache
            agent_cache[cache_key] = (graph_agent, recomm_agent)

        config = {"configurable": {"thread_id": cache_key}, "recursion_limit": 15}
        assistant_response = await graph_agent.ainvoke(
            AgentState(messages=[HumanMessage(role="user", content=command_text)], rag_evaluation=None), config=config
        )

        hist_service.add_message(
            user_id=user_id, role="assistant", content=assistant_response["messages"][-1].content
        )
        return ChatResponse(response=assistant_response["messages"][-1].content)

    except Exception as e:
        logger.error(f"Error processing user command for user_id={user_id}: {e}", exc_info=True)
        return ChatResponse(error="An internal error occurred while processing your request.")


# -----------------------------------------------------------------------------
# Development/Testing Entry Point
# -----------------------------------------------------------------------------
# This section is for local development testing only.
# Run with: python -m backend.services.conversation

if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    async def _test_conversation():
        """Test function for local development."""
        from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model
        embedding_model, _ = load_embedding_model()

        hist_service.delete_history(user_id=1)

        res = await process_user_command(
            user_id=1,
            db_name="user-1",
            command_text="In the photo at the stadium a few months ago, what color were the stands?",
            embedding_model=embedding_model,
        )
        print(res.response)

    asyncio.run(_test_conversation())
