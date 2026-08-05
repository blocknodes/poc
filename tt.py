import asyncio
import websockets
import json
import sys

WS_URL = "ws://10.18.210.7:32018/v1.0/test"
WAIT_ALL_MSG_TIMEOUT = 3000

def parse_escaped_json(obj):
    """递归解析所有内嵌转义JSON字符串"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_dict[k] = parse_escaped_json(v)
        return new_dict
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            new_list.append(parse_escaped_json(item))
        return new_list
    elif isinstance(obj, str):
        try:
            loaded = json.loads(obj)
            return parse_escaped_json(loaded)
        except (json.JSONDecodeError, ValueError):
            return obj
    else:
        return obj

async def ws_client(retext_content, mode="single"):
    send_data = {
        "feature_code": "861003009000014000000712",
        "device_id": "86100300900001400000071212345678",
        "retext": retext_content,
        "client_sid": "A06212345678,1723550666666",
        "tv_mode": "0",
        "enable_slow": True
    }
    result = {
        "success": False,
        "recv_mode": mode,
        "send_data": send_data,
        "all_response": [],
        "error": None
    }
    try:
        async with websockets.connect(WS_URL) as websocket:
            send_str = json.dumps(send_data, ensure_ascii=False)
            await websocket.send(send_str)

            if mode == "single":
                # 只接收单条消息
                raw = await websocket.recv()
                raw_json = json.loads(raw)
                clean_json = parse_escaped_json(raw_json)
                result["all_response"].append(clean_json)
            else:
                # 持续接收全部消息
                async def recv_loop():
                    while True:
                        raw = await websocket.recv()
                        raw_json = json.loads(raw)
                        clean_json = parse_escaped_json(raw_json)
                        result["all_response"].append(clean_json)
                await asyncio.wait_for(recv_loop(), timeout=WAIT_ALL_MSG_TIMEOUT / 1000)

            result["success"] = True
    except asyncio.TimeoutError:
        result["success"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 解析入参
    retext_arg = "播放流浪地球"
    recv_mode_arg = "single"

    if len(sys.argv) >= 2:
        retext_arg = sys.argv[1]
    if len(sys.argv) >= 3:
        recv_mode_arg = sys.argv[2].lower()

    asyncio.run(ws_client(retext_arg, recv_mode_arg))
