import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

TMP_DIR = os.getenv("TMP_DIR")

EXTRACTION_MODEL_NAME = os.getenv("EXTRACTION_MODEL_NAME")
EXTRACTION_LLM_CONFIG = os.getenv("EXTRACTION_LLM_CONFIG")
AGENT_LLM_CONFIG = os.getenv("AGENT_LLM_CONFIG")
CHATBOT_LLM_CONFIG = os.getenv("CHATBOT_LLM_CONFIG")
CHATBOT_EMBEDDING_CONFIG = os.getenv("CHATBOT_EMBEDDING_CONFIG")
IMG_ANALYSIS_VL_CONFIG = os.getenv("IMG_ANALYSIS_VL_CONFIG")

IMG_DEEP_ANALYSIS_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant that selects a list of photos that could be useful to answer the question.

---Goal---

Generate a response that contains a list of up to 3 photos that could be useful to answer the question if analyzed in-depth, using the context provided in the Data Tables.
You must reply only with the list of photos and nothing else, avoid other unnecessary comments.

The photo entities in the Data Tables have a name in the following format: photo_id.
If there are fewer than 3 "photo" entities that can be useful for the question, list fewer.

If there are no photos useful for the question, reply "No photo matches the question".

---Examples---
"user": "Was I wearing blue shoes to Kevin's birthday?
"assistant": "- photo_5231
- photo_7343
- photo_5241"

---Data tables---

{context_data}


---Target response length and format---

{response_type}

"""

QUERY_TO_RESOLVE_IMG_NAME = """
MATCH (d:Document {fileName: $photo_id + '.txt'})<-[:PART_OF]-(c:Chunk)
WHERE NOT c.fileName ENDS WITH '.txt'
RETURN c.fileName
"""

QUERY_TO_MERGE_KGs = """
MATCH (d1:Document {fileName: $fileName})
MATCH (d2:Document {fileName: $new_fileName_u})
MATCH (c:Chunk)-[r]-(d2)
MERGE (c)-[:PART_OF]->(d1)
DELETE r, d2;
"""

QUERY_TO_CONNECT_TO_USER = """
MATCH (u:User)
MATCH (e:__Entity__ {id: $fileName})
CALL apoc.create.relationship(u, $relation, {}, e)
YIELD rel
RETURN rel
"""

QUERY_TO_DELETE_USER_CONNECTION = """
MATCH (u:User)-[r]->(e:__Entity__ {id: $fileName})
DELETE r
"""

ENTITIES_QUERY = """
MATCH (e:__Entity__)
WITH e, 
    [comm IN e.communities | toString(comm) + '0'] AS community_ids

OPTIONAL MATCH (e)-[r]-()
WITH e, community_ids,
    count(r) AS deg

OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(ch:Chunk)
WITH e, community_ids, deg,
    collect(elementId(ch)) AS chunk_ids,
    [lbl IN labels(e) WHERE lbl <> "__Entity__"][0] AS type

WITH collect({
    e: e,
    community_ids: community_ids,
    deg: deg,
    chunk_ids: chunk_ids,
    type: type
}) AS rows

UNWIND range(0, size(rows)-1) AS i
WITH i + 1 AS human_readable_id, rows[i] AS row

RETURN
    row.e.id AS title,
    row.type AS type,
    row.e.id AS description,
    row.deg AS degree,
    human_readable_id,
    elementId(row.e) AS id,
    row.chunk_ids AS text_unit_ids,
    row.community_ids AS communities,
    row.e.embedding AS name_embedding,
    row.e.embedding AS description_embedding,
    size(row.chunk_ids) AS frequency
"""

RELATIONSHIPS_QUERY = """
MATCH (s:__Entity__)-[r]->(t:__Entity__)
WITH s, t, r

OPTIONAL MATCH (s)-[r1]-()
WITH s, r, t, 
    count(DISTINCT (r1)) AS r1

OPTIONAL MATCH (t)-[r2]-()
WITH s, t, r, r1,
    count(DISTINCT (r2)) AS r2

OPTIONAL MATCH (t)-[]-(c1:Chunk)
OPTIONAL MATCH (s)-[]-(c2:Chunk)
WITH s, t, r, r1, r2,
    toInteger(r1) + toInteger(r2) AS comb_deg,
    collect(DISTINCT elementId(c1)) + collect(DISTINCT elementId(c2)) AS text_unit_ids

WITH collect({s:s, t:t, r:r, comb_deg:comb_deg, text_unit_ids:text_unit_ids}) AS rows

UNWIND range(0, size(rows)-1) AS i
WITH i + 1 AS human_readable_id, rows[i] AS row

RETURN 
    row.s.id AS source,
    row.t.id AS target,
    elementId(row.s) AS source_id,
    elementId(row.t) AS target_id,
    toString(1.0) AS weight,
    row.s.id + '_' + type(row.r) + '_' + row.t.id AS description,
    row.text_unit_ids AS text_unit_ids,
    human_readable_id,
    row.comb_deg AS combined_degree,
    elementId(row.r) AS id
"""

TEXT_UNIT_QUERY = """
MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
WITH c,
    collect(DISTINCT elementId(e)) AS entity_ids

OPTIONAL MATCH (c)-[r]-()
WITH c, entity_ids,
    collect(DISTINCT elementId(r)) AS relationships_ids

OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
WITH c, entity_ids, relationships_ids,
    collect(DISTINCT elementId(d)) AS document_ids

RETURN
    document_ids AS document_id,
    elementId(c) AS id,
    c.text AS text,
    entity_ids AS entity_ids,
    relationships_ids AS relationships_ids
"""


COMMUNITIES_QUERY = """
MATCH (c:__Community__)
OPTIONAL MATCH (c)-[:PARENT_COMMUNITY]->(p:__Community__)
WITH
  c, p,
  toInteger(split(c.id,'-')[1] + c.level) AS community_uid,
  CASE
    WHEN p IS NULL THEN -1
    ELSE toInteger(split(p.id,'-')[1] + p.level)
  END AS parent_uid

// children
OPTIONAL MATCH (c)<-[:PARENT_COMMUNITY]-(child:__Community__)
WITH
  c, p, community_uid, parent_uid,
  [x IN collect(DISTINCT child) WHERE x IS NOT NULL |
    toInteger(split(x.id,'-')[1] + x.level)
  ] AS children_uids

// entities in community
OPTIONAL MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
WITH
  c, p, community_uid, parent_uid, children_uids,
  collect(DISTINCT toString(elementId(e))) AS entity_element_ids,
  collect(DISTINCT e) AS member_nodes,
  count(DISTINCT e) AS size

// relationships among member entities
OPTIONAL MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
WHERE e1 IN member_nodes AND e2 IN member_nodes
WITH
  c, p, community_uid, parent_uid, children_uids, entity_element_ids, size, member_nodes,
  collect(DISTINCT toString(elementId(r))) AS relationship_ids

// chunks pointing to member entities
OPTIONAL MATCH (e3:__Entity__)<-[r1:HAS_ENTITY]-(ch:Chunk)
WHERE e3 IN member_nodes
WITH
  c, p, community_uid, parent_uid, children_uids, entity_element_ids, size, relationship_ids,
  collect(DISTINCT toString(elementId(ch))) AS chunk_ids

RETURN
  elementId(c)                     AS id,
  community_uid                    AS community,
  community_uid                    AS human_readable_id,
  toInteger(c.level)               AS level,
  parent_uid                       AS parent,
  children_uids                    AS children,
  c.title                          AS title,
  entity_element_ids               AS entity_ids,
  relationship_ids                 AS relationship_ids,
  chunk_ids                        AS text_unit_ids,
  toString(date())                 AS period,
  toInteger(size)                  AS size
ORDER BY community ASC;
"""
