import asyncio

import websockets

active_connections = set()


async def handler(websocket, path):
    # Add newly connected client to the set of active connections
    active_connections.add(websocket)
    try:
        while True:  # Infinite loop to keep the server running
            payload = await websocket.recv()
            print(f"Received payload: {payload}")
            if websocket.open:  # Check if the connection is open
                # Broadcast the payload to all active connections
                for conn in active_connections:
                    if conn != websocket and conn.open:
                        await conn.send(payload)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection is closed.")
    finally:
        # Remove the client from the set of active connections
        active_connections.remove(websocket)


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server started at ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
