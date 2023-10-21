# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from llama import Llama, Dialog
import http.server
import socketserver
import json

ckpt_dir = "llama-2-7b-chat/"
tokenizer_path = "tokenizer.model"
max_seq_len = 512
max_batch_size = 6
temperature: float = 0.6
top_p: float = 0.9
max_gen_len: Optional[int] = None


class MyHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, response_content):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_content).encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        try:
            json_data = json.loads(post_data)
            request = json_data["request"]
            response = get_response(request)
            response = {"response": response}
            self._send_response(200, response)
        except ValueError as e:
            error_response = {"error": "Invalid JSON data"}
            self._send_response(400, error_response)


class LlamaGenerator:
    generator: Llama

    def generate(self):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )


def get_response(request="hello"):
    dialogs: List[Dialog] = [
        [{"role": "user", "content": request}],
    ]

    print("> user: " + request)

    results = llamaGenerator.generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for result in results:
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
    print("\n==================================\n")

    if len(results) > 0:
        return results[0]['generation']['content']

    return "I do not understand the request!!!"


llamaGenerator = LlamaGenerator()
llamaGenerator.generate()

# Set the server address and port
server_address = ('', 8000)

# Create and start the server
httpd = socketserver.TCPServer(server_address, MyHandler)
print("Server is running on port 8000...")
httpd.serve_forever()
