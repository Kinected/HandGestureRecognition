import asyncio
import json

import websockets


async def connect(uri):
    while True:
        try:
            print("Connecting to server...")
            websocket = await websockets.connect(uri)
            print("Connected to server.")
            return websocket
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            print("Retrying in 1 seconds...")
            await asyncio.sleep(1)  # Attendre avant de tenter de se reconnecter


async def send_gesture(websocket, gesture: dict):
    await websocket.send(json.dumps(gesture))
