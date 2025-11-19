#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, urlencode
import webbrowser

# ====== CONFIG ======
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
]
REDIRECT_URI = "http://localhost:8765/callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
PORT = 8765

CLIENT_SECRET_FILE = "client_secret.json"  # scaricato da Google Cloud Console

def load_client_id():
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise RuntimeError(f"{CLIENT_SECRET_FILE} non trovato.")
    with open(CLIENT_SECRET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    info = data.get("installed") or data.get("web") or {}
    client_id = info.get("client_id")
    if not client_id:
        raise RuntimeError("client_id mancante nel file di credenziali.")
    return client_id


# ====== HTTP handler to capture ?code ======
class CallbackHandler(BaseHTTPRequestHandler):
    auth_code = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        qs = parse_qs(parsed.query)
        CallbackHandler.auth_code = (qs.get("code") or [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<h1>OK!</h1><p>Accesso completato. Puoi chiudere questa finestra e tornare al terminale.</p>"
        )
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, fmt, *args):
        return  # silenzia log

def main():
    client_id = load_client_id()
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = f"{AUTH_URL}?{urlencode(params)}"

    print("Aprendo il browser per l'autenticazione Google...")
    webbrowser.open(url)

    httpd = HTTPServer(("127.0.0.1", PORT), CallbackHandler)
    httpd.serve_forever()

    if not CallbackHandler.auth_code:
        raise RuntimeError("Nessun 'code' ricevuto dal redirect.")

    print("\n================ AUTH CODE =================")
    print(CallbackHandler.auth_code)
    print("===========================================\n")

if __name__ == "__main__":
    main()
