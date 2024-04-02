import json

from websockets.exceptions import ConnectionClosedError


async def send_gesture(websocket, gesture: dict):
    try:
        await websocket.send(json.dumps(gesture))
    except ConnectionClosedError:
        print("WebSocket connection was closed unexpectedly.")
