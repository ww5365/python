from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from ddgs import DDGS

import os
import datetime
from typing import List, Dict, Any

# 设置环境变量（可选）
os.environ["OPENAI_API_KEY"]=

os.environ["DDG_USER_AGENT"] = "Mozilla/5.0"
os.environ["REQUESTS_CA_BUNDLE"] = ""  # 禁用 SSL 验证（仅用于测试）

# 初始化 LLM（你已提供）
llm = init_chat_model(model= "openai:gpt-5-mini")


# 定义工具函数
@tool
def web_search(query: str) -> str:
    """使用 DDGS 搜索指定的查询内容。

    Args:
        query: 要搜索的关键词或问题
    """
    with DDGS(verify=False) as ddgs:
        results = ddgs.text(query, max_results=10)
        return "\n".join([f"{result['title']}: {result['href']}" for result in results])

# 将 LLM 与工具绑定
llm_with_tools = llm.bind_tools([web_search])

# 工具映射
tools_by_name = {"web_search": web_search}

# 获取今天的日期
today = datetime.date.today()
dates_str = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]

# LLM 节点：决定是否调用工具
def llm_call(state: MessagesState):
    """LLM 决定是否调用工具"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="你是一个技术资讯获取和整理助手。"
                    )
                ]
                + state["messages"]
            )
        ]
    }

# 工具调用节点
def tool_node(state: dict):
    """执行工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        print(f'调用工具: {tool_call["name"]}({tool_call["args"]}) -> {observation}')
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 信息评估与摘要生成节点（真实实现）
def evaluate_and_summarize(state: dict):
    """评估信息价值并生成 Top 10 摘要"""
    # 提取搜索结果
    search_results = state["messages"][-1].content
    # 使用 LLM 评估并生成摘要
    summary = llm.invoke(f"请评估以下搜索结果，进行整理(去重、合并、优选等)并生成摘要，按地区分中国和美国各整理Top 10。要求如下："
                         f"1. 按权威性大厂(如：OpenAI、Google、Apple、阿里、字节、Coze、LangGraph、Dify、n8n)相关的靠前"
                         f"2. 同时参考相关性、流量、最新性等"
                         f"3. 输出中文"
                         f"4. 同时列出来源链接、发布时间、关键词标签等：{search_results}")
    # 获取上一个 AIMessage 的 tool_calls（遍历找到最近的 AIMessage）
    tool_call_id = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_id = msg.tool_calls[0]["id"]
            break
    return {"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id, name="summary")]}

# 条件判断函数：是否继续执行
def should_continue(state: MessagesState) -> str:
    """根据是否调用工具决定是否继续"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# 构建状态图
def assistant_agent():
    """构建并返回每日讯息助手 Agent"""
    agent_builder = StateGraph(MessagesState)

    # 添加节点
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("evaluate_and_summarize", evaluate_and_summarize)

    # 添加边
    agent_builder.add_edge("tool_node", "evaluate_and_summarize")
    agent_builder.add_edge("evaluate_and_summarize", END)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

    # 编译 agent
    agent = agent_builder.compile()
    return agent


def print_summary_pretty(summary: str):
    """将摘要内容按段落打印，每段之间用空行分隔"""
    paragraphs = summary.split("\n\n")
    for paragraph in paragraphs:
        print(paragraph)
        print()  # 添加空行


# 主函数入口
if __name__ == "__main__":
    # 构建 agent
    agent = assistant_agent()

    # 可视化 agent 图（可选）
    # print(agent.get_graph(xray=True).draw_ascii())

    # 构造输入消息
    messages = [HumanMessage(content=f"搜索并评估 {dates_str}发布的 Agent 智能体技术，包括中国国内和海外地区。整理(包括去重、合并、优选等) ，生成中国和海外的各自Top 10文章")]

    # 调用 agent
    result = agent.invoke({"messages": messages})

    # 输出结果
    print("\n--- Agent 讯息摘要 ---")
    # for msg in result["messages"]:
    #     print(msg)
    summary = result["messages"][-1].content
    print(summary.replace("\\n\\n", "\n\n").replace("\\n", "\n"))

