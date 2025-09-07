from tools_local import LocalSearchTool, ReaderTool
ls = LocalSearchTool()
print(ls.forward(query="PPO vs TRPO", k=3))  # should return {"results": [...]}

rd = ReaderTool()
first = ls.forward(query="PPO vs TRPO", k=1)["results"][0]["chunk_id"]
print(rd.forward(chunk_id=first))            # should return text + citation
