import asyncio
import os
import json
from typing import Optional, List
from contextlib import AsyncExitStack
import re
from openai import OpenAI 
from dotenv import load_dotenv
from utility import parse_json_response,extract_all_json_strings

from search_tools import multi_search
from reasoning_tools import deep_reason

load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(load_dotenv(dotenv_path=dotenv_path))

class MCPClient:

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

    async def eval_llm_response(self,query_list:list[str],context:str):
        prompt = f"""You are an intelligent query optimizer. Your task is to evaluate whether the provided context can sufficiently answer a given query, and if not, generate one additional query to complement the existing query list for better information retrieval.

            **Input:**
            - Query to evaluate: "{query_list[0]}"
            - Existing query list: {query_list}
            - Context (in JSON format): {context}

            **Evaluation Criteria:**
            1. The context must contain **direct, specific, and complete** information to fully answer the query.
            2. Partial, vague, or irrelevant information means the context is insufficient.
            3. The answer must not require external knowledge or assumptions.

            **Output Rules (STRICT FORMAT REQUIRED):**
            - If the context CAN sufficiently answer the query: return `{{"new_query": ""}}`
            - If the context CANNOT sufficiently answer the query: return `{{"new_query": "a concise, focused, and information-seeking query that complements the existing query list"}}`

            **Important:**
            - Return ONLY a valid JSON object with the key `new_query`.
            - The value must be a string, or an empty string if no new query is needed.
            - Do not include any explanations, markdown, or extra text.
            - Ensure the new query avoids redundancy with the existing query list."""
                    

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5000
        )


        response_json = parse_json_response(response.choices[0].message.content.strip())
        new_query = response_json.get("new_query", "")
        if not isinstance(new_query, str):
            new_query = str(new_query)
        
        return new_query.strip()
    
    async def mutual_call(self, query: str,tool_plan:list,iterations:int)-> str:
        iter_result={}       
        iter_result['iteration']=iterations
        for step in tool_plan:
            tool_name = step["name"]

            if tool_name == "multi_search":
                iter_result["multi_search"]=multi_search(query,iter_result,n_wide=3 )
            elif tool_name=="deep_reason" :
                iter_result["deep_reason"]=deep_reason(query, iter_result,n_deep=3)
               
        json_result = json.dumps(iter_result, ensure_ascii=False, indent=2)

        return json_result


    async def process_query(self, query: str) -> str:

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "multi_search",
                      "description":"Financial research tool that uses LLM to generate diversified search queries from a core topic, executes parallel web searches via Brave API (with Serper fallback), aggregates and deduplicates results, and returns structured JSON with search metadata including original query, sub-queries, and results."

                }
            } ,
            {
                "type": "function",
                "function": {
                    "name": "deep_reason",
                    "description": "Generates novel, multi-directional analytical insights by splitting a user query into n_deep distinct reasoning angles, performing parallel deep LLM analysis on each angle to uncover non-obvious causal mechanisms and contrarian perspectives, and synthesizing breakthrough insights into a structured markdown report with sectioned analysis and strategic depth. Prioritizes novel, insight-driven conclusions beyond surface-level analysis using contextual inputs (knowledge graphs, historical data) to produce high-value strategic insights."
                }
            }
 
        ]



        tool_plan = await self.plan_tool_usage(query, available_tools)

        # 依次执行工具调用，并收集结果
        iterations=1
        max_iter=5
        query_list=[query]
        all_iterations=[]
            # result = await self.session.call_tool(tool_name, tool_args)

            # tool_outputs[tool_name] = result.content[0].text
            # messages.append({
            #     "role": "tool",
            #     "tool_call_id": tool_name,
            #     "content": result.content[0].text
            # })
        json_result=await self.mutual_call(query,tool_plan,iterations)
        all_iterations.append(json_result)
        while iterations<max_iter :
            new_result_query= await self.eval_llm_response(query_list,json_result)
            if new_result_query:
                print("Added new query: {}".format(new_result_query))
                query_list.append(new_result_query)
                iterations=iterations+1
                tool_plan = await self.plan_tool_usage(new_result_query, available_tools)
                json_result=await self.mutual_call(new_result_query,tool_plan,iterations)
                all_iterations.append(json_result)
            else:
                print("It is enough for current queries")
                break

        
        #总结所有回答
        
        prompt=f"""You are an intelligent answer synthesizer. Your task is to generate a clear, accurate, specific, and comprehensive response to the user's query based on the provided multi-round retrieval and reasoning results.

                **Input:**
                - User Query: "{query_list[0]}"
                - Retrieval & Reasoning History (in List[JSON] format): 
                {all_iterations}

                **Instructions:**
                1. **Comprehensively analyze** all iterations, especially the final `deep_reason_result` in the last round.
                2. **Synthesize information** from all `multi_search_result` and `deep_reason_result` entries.
                3. Focus on **relevance, accuracy, and completeness** — do not include unsupported claims.
                4. If the information is sufficient, provide a well-structured final answer.
                5. If the information is still insufficient, state: "The available information is not sufficient to provide a complete answer."

                **Output Format (STRICT):**
                - Return only a JSON object with the key `"final_answer"`.
                - The value must be a string.
                - Do not include any explanations, markdown, or extra text.

                Example Output:
                {{"final_answer": "The A350-1000 has a range of approximately 15,000 km, powered by Rolls-Royce Trent XWB engines, with a typical seating capacity of 366 passengers in a three-class configuration."}}

                Now generate the answer:
                        """


        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5000
        )


        resp_json = parse_json_response(resp.choices[0].message.content.strip())
        final_answer = resp_json.get("final_answer","")


        # # 调用大模型生成回复信息，并输出保存结果
        # final_response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=messages
        # )
        # final_output = final_response.choices[0].message.content

        return final_answer

    async def chat_loop(self):
        # 初始化提示信息
        print("\n🤖 MCP 客户端已启动！输入 'quit' 退出")

        # 进入主循环中等待用户输入
        while True:
            try:
                query = input("\n你: ").strip()


                if query.lower() == 'quit':
                    break

                #
                # 处理用户的提问，并返回结果
                
                response = await self.process_query(query)
                print(f"\n🤖 AI: {response}")

            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def plan_tool_usage(self, query: str, tools: List[dict]) -> List[dict]:
        # 构造系统提示词 system_prompt。
        # 将所有可用工具组织为文本列表插入提示中，并明确指出工具名，
        # 限定返回格式是 JSON，防止其输出错误格式的数据。
        print("\n📤 提交给大模型的工具定义:")
        print(json.dumps(tools, ensure_ascii=False, indent=2))
        tool_list_text = "\n".join([
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in tools
        ])
        system_prompt = {
    "role": "system",
    "content": (
        "You are an intelligent task planning assistant. The user will provide a natural language request.\n"
        "You MUST select tools ONLY from the following list (use the exact tool names as given):\n"
        f"{tool_list_text}\n"
        "If multiple tools are required in sequence, you may reference the output of a previous step using the placeholder {{tool_name}} in the arguments of a subsequent step.\n"
        "Your response MUST be a valid JSON array. Each object in the array must contain at  a \"name\" field (string) with the exact tool name.\n"
        "DO NOT include any natural language, explanations, markdown, or tools not listed above. Return ONLY the JSON array."
    )
}

        # 构造对话上下文并调用模型。
        # 将系统提示和用户的自然语言一起作为消息输入，并选用当前的模型。
        planning_messages = [
            system_prompt,
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=planning_messages,
            tools=tools,
            tool_choice="none"
        )

        # 提取出模型返回的 JSON 内容
        content = response.choices[0].message.content.strip()
        # match = re.search(r"```(?:json)?\\s*([\s\S]+?)\\s*```", content)
        # if match:
        #     json_text = match.group(1)
        # else:
        #     json_text = content

        plan = extract_all_json_strings(content)
        plan = json.loads(plan[0])
        print("selected tool name:{}".format(plan))

        return plan

        # 在解析 JSON 之后返回调用计划
        # try:
        #     plan = json.loads(json_text)
        #     return plan if isinstance(plan, list) else []
        # except Exception as e:
        #     print(f"❌ 工具调用链规划失败: {e}\n原始返回: {content}")
        #     return []

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    # server_script_path = "D:\\mcp-project\\server.py"
    client = MCPClient()
    try:
        # await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

