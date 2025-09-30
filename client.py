import asyncio
import os
import json
from typing import Optional, List
from contextlib import AsyncExitStack
from openai import OpenAI 
from dotenv import load_dotenv
from utility import parse_json_response,extract_all_json_strings, count_meaningful_chars

from search_tools import multi_search
from deep_reasoning import deep_reason
# from reasoning_tools import deep_reason


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


    async def compress_str(self,context:List[str]):
        prompt=f"""You are a financial semantic text compressor for global equity research. Your task is to extract and distill the core market-relevant meaning from deeply nested, escaped JSON strings into a single, coherent, and highly compressed plain text paragraph suitable for **investment analysis, risk assessment, or portfolio decision-making**.

        ## Input Format
        You will receive one or more JSON-like strings with:
        - Outer fields: "iteration", "multi_search", "deep_reason"
        - Inner values of "multi_search" and "deep_reason" are **escaped JSON strings**, 
        - Content includes search queries, web snippets, summaries, and analytical notes on corporate events

        ## Your Task
        1. **Parse** the input correctly — extract and decode escaped JSON content.
        2. **Identify and preserve only financially material information**, such as:
        - Product issues affecting brand equity, recall cost, or liability exposure
        - Regulatory changes impacting compliance cost, market access, or competitive moat
        - Customer sentiment shifts influencing conversion, retention, or NPS
        - Strategic initiatives (pricing, OTA, customization) affecting margins or differentiation
        - Executive or governance events with reputational or operational risk
        3. **Remove all of the following**:
        - Structural syntax: {{}}, `[]`, `:`, `"`, `\`, `\n`, `\t`
        - Boilerplate: e.g., "Stock tips from Jinqilin Analyst Reports", "Published in XX", "Source: XXX", "Official account"
        - URLs, metadata fields (e.g., "link", "source_query_idx")
        - Redundant or near-identical statements — keep only the most complete and material version
        - Vague or promotional language: e.g., "in-depth study", "breaking news", "game-changing", "authoritative, timely"
        4. **Merge related financial signals** into concise, professional sentences. For example:
        - Multiple points about software recalls → one sentence on scale, cause, and **impact on brand trust and potential warranty liability**
        - Repeated claims about narrative-driven growth → one sentence on **narrative dependency increasing sentiment volatility and conversion risk**
        5. **Output Requirements**:
        - One continuous paragraph in **clear, neutral, professional English**
        - No bullet points, headings, markdown, or JSON
        - No prefixes like "Summary:" or explanations
        - Use only essential punctuation (periods, commas)
        - Maximize information density while preserving key financial implications

        ## Financial Analysis Example Handling (Follow These Patterns)

        ### Example 1: Product Risk & Regulatory Exposure
        Input: `"title": "Tesla recalls 2 million vehicles over Autopilot vision-only system safety concerns"`
        → Output integration:  
        "Tesla recalled 2M vehicles over Autopilot's vision-only design, highlighting regulatory scrutiny and potential liability exposure; NHTSA's evolving ADAS rules may increase validation costs and delay FSD monetization."

        ### Example 2: Sentiment Volatility & Conversion Risk
        Input: `"summary": "Apple's 'crisis of innovation' narrative resurfaces after mixed App Store growth and Vision Pro adoption"`
        → Output integration:  
        "Resurgence of 'innovation stagnation' narrative pressures Apple's premium valuation, increasing churn risk in growth markets and weakening developer ecosystem stickiness."

        ### Example 3: Monetization Strategy & Margin Outlook
        Input: `"bullets": ["Meta increases ad load in Stories and Reels", "User engagement drops 12% in test markets"]`
        → Output integration:  
        "Higher ad load in Meta's short-form video correlates with engagement decline, threatening long-term user retention and reducing CPM sustainability."

        ### Example 4: Governance & Reputational Risk
        Input: `"title": "Volkswagen CFO resigns amid emissions scandal compliance review"`
        → Output integration:  
        "CFO resignation during ongoing emissions compliance audit signals governance instability, potentially triggering credit rating scrutiny and investor outflows."

        ### Example 5: Regulatory Tailwinds & Competitive Advantage
        Input: `"deep_reason": "{{"insight\": \"EU AI Act will favor firms with auditable training data and transparent model governance.\"}}"
        → Output integration:  
        "EU AI Act creates structural advantage for firms with documented data lineage and model governance, benefiting incumbents with mature compliance infrastructure."

        ## Final Output Example (Finance-Only, Global Focus)
        Tesla recalled 2M vehicles over Autopilot's sensor design, signaling regulatory and liability risks under tightening NHTSA rules. Apple faces valuation pressure from renewed 'innovation stagnation' concerns and weak Vision Pro uptake. Meta's ad load increase risks user churn, threatening CPM stability. Volkswagen's CFO exit amid compliance review raises governance red flags. Firms with auditable AI systems gain advantage under EU AI Act, reinforcing moat potential in regulated markets.

        Now process the following input:
        {context}

        Return only the compressed plain text:"""
        
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=5000
    )

        response_text = response.choices[0].message.content.strip()

        return response_text




    async def eval_llm_tool_useage(self,query_list:list[str],context:str,tools_list: List[dict]):
        prompt = f"""You are an intelligent evaluator and tool planner. The user provides:
        - A primary query to answer: {query_list[0]}
        - A list of queries already used for retrieval: {query_list}
        - Retrieved context (in JSON format): {context}
        - A list of available tools you may use: {tools_list}

        Your task is to decide whether the current context is sufficient to fully and unambiguously answer the primary query without assumptions.

        AVAILABLE TOOLS (you MUST choose ONLY from these, using exact names):
        {tools_list}

        RESPONSE RULES:
        1. If the context IS sufficient:
        - Set "query" to an empty string.
        - Set "toolname" to an empty list.
        2. If the context IS NOT sufficient:
        - Generate ONE concise, focused, non-redundant query that captures the missing information.
        - List ALL tools (as objects with "name" key) needed to answer that query. You may include multiple tools.
        - Each tool must appear as: {{"name": "exact_tool_name"}}.
        3. NEVER invent tool names—only use those listed above.
        4. NEVER include explanations, markdown, code blocks, extra fields, natural language, or formatting.
        5. The new query must be answerable by the selected tools and avoid overlap with the existing query list.

        OUTPUT FORMAT:
        - Your entire response must be a single-line, valid JSON object.
        - NO pretty-printing, NO newlines, NO indentation, NO backticks, NO "```json", NO comments.
        - Example sufficient: {{"query": "", "toolname": []}}
        - Example insufficient: {{"query": "How is the btc price", "toolname": [{{"name": "multi-search"}}, {{"name": "deep_reason"}}]}}

        Return ONLY the JSON object. Do not say anything else. Do not wrap it in anything."""
    
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5000
        )

        response_text = response.choices[0].message.content.strip()
        try:
                data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")

        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        if "query" not in data or "toolname" not in data:
            raise ValueError("Missing required fields: 'query' and 'toolname'")
        if not isinstance(data["toolname"], list):
            raise ValueError("'toolname' must be a JSON array")

        query = data["query"]
        toolname = data["toolname"]

        # If query is empty, return (None, None) to indicate no action needed
        if not query:
            return None, None

        if toolname:
            print("selected tool name:{}".format(toolname))

        return query, toolname

        
    async def eval_llm_response(self,query_list:list[str],context:str):
        prompt = f"""You are an intelligent query optimizer. Your task is to evaluate whether the provided context can sufficiently answer a given query, and if not, generate one additional query to complement the existing query list for better information retrieval.

            **Input:**
            - Query to evaluate: {query_list[0]}
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
        char_count = 0
            # result = await self.session.call_tool(tool_name, tool_args)

            # tool_outputs[tool_name] = result.content[0].text
            # messages.append({
            #     "role": "tool",
            #     "tool_call_id": tool_name,
            #     "content": result.content[0].text
            # })
        json_result=await self.mutual_call(query,tool_plan,iterations)
        all_iterations.append(json_result)
        char_count = char_count +count_meaningful_chars(json_result)
        compress_context =[]
        while iterations<max_iter :
            new_result_query, tool_plan = await self.eval_llm_tool_useage(query_list,all_iterations,available_tools)
            print("iteration: {}".format(iterations))
            if new_result_query:
                print("Added new query: {}".format(new_result_query))
                query_list.append(new_result_query)
                iterations=iterations+1
                if tool_plan:
                    json_result=await self.mutual_call(new_result_query,tool_plan,iterations)
                    all_iterations.append(json_result)
                    char_count = char_count +count_meaningful_chars(json_result)
                    if char_count >100000:
                       new_context = await self.compress_str(all_iterations)
                       compress_context.append(new_context)
                       all_iterations=[]
                       char_count=0

                else :
                    print("we have new query, but no toolname")
            else:
                print("It is enough for current queries")
                break

        all_iterations.extend(compress_context)
        #总结所有回答
        print(" We are reporting")
       

        prompt=f"""You are a senior financial analyst responsible for synthesizing multi-source research into a professional, evidence-based analysis. Your output will be used in investment memos, client reports, and internal strategy discussions.

        **User Query:**  
        "{query_list[0]}"

        **Retrieval & Reasoning History (List[JSON]):**  
        {all_iterations}

        **Your Task:**  
        Based on the full retrieval and reasoning history, produce a **comprehensive, detailed, and professionally written financial analysis** that fully leverages all available information.

        **Instructions:**               

        1. **Analyze all rounds thoroughly**, **Comprehensively analyze** all iterations, not just the final one. Pay close attention to:
        - **Synthesize information** from all `multi_search` and `deep_reason` entries.
        - Qualitative insights and logical inferences from `deep_reason_result`: competitive dynamics, regulatory risks, management strategy, macroeconomic impacts, technological shifts
        - Evolution of understanding: Did early results suggest one conclusion, but later evidence refine or contradict it? Explain briefly.
        - Focus on **relevance, accuracy, and completeness** — do not include unsupported claims.
        - Specific examples, company names, analyst quotes, or source references (e.g., "per Goldman Sachs' April 2025 report")

        2. **Your response must include:**
        - A clear, direct answer to the user's query in the opening sentence(s)
        - **Specific financial figures** with context (e.g., "Amazon's AWS revenue grew 17% YoY to $25.3 billion in Q1 2025, slightly below consensus of 18%")
        - **Real-world examples** (e.g., "This trend is evident in JPMorgan's recent acquisition of a fintech startup to enhance its digital lending platform")
        - **Comparative analysis** (e.g., "Compared to the sector median EV/EBITDA of 10.5x, Company X trades at 13.2x, reflecting premium growth expectations")
        - Where applicable, explain how the conclusion was reached through iterative reasoning

        3. **Tone and Style:**
        - Write in a professional, objective style
        - Avoid markdown, bullet points, JSON, or any formatting
        - Do NOT include phrases like "the data shows" or "based on the results"—state conclusions directly and confidently if supported
        - If the information is still insufficient, state: "The available information is not sufficient to provide a complete answer."

        **Output Format:**
        - Return **only the plain text analysis**, with no prefixes (e.g., "Answer:", "Final Response:"), no quotes, no JSON
        - Respond **in the same language as the user's query**
        - Ensure total length is **approximately 1000 Chinese characters** (for Chinese) or **1000 words** (for English)
        
        **Example Output:**  
        Tesla's vehicle delivery growth slowed to 6% year-over-year in Q1 2025, totaling 440,000 units, missing analyst expectations of 460,000. This reflects ongoing pricing pressure in key markets like China and Europe, where average selling prices declined by 8–10% to maintain volume. However, energy storage deployments surged to 10.5 GWh, up 45% YoY, driven by Megapack demand from utility-scale projects in the U.S. and Australia. Gross margin improved to 19.8% from 18.2% in Q4 2024, primarily due to lower battery costs and factory efficiency gains at Giga Texas. Despite delivery headwinds, recurring software revenue from Full Self-Driving (FSD) subscriptions reached $280 million annualized, growing at 30% CAGR. Analysts at Morgan Stanley note that FSD could contribute over $10 billion in revenue by 2027 if regulatory approval expands. Valuation remains elevated at 65x forward P/E, but this is supported by long-term AI and robotics optionality. In summary, Tesla is transitioning from a pure EV play to a broader AI and energy technology company, though near-term execution risks persist in the automotive segment.

        Now generate the final financial analysis:
        """
        



        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=12000
        )

        # resp_json = parse_json_response(resp.choices[0].message.content.strip())
        # final_answer = resp_json.get("final_answer","")
        final_answer = resp.choices[0].message.content.strip()

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
    "List ALL tools (as objects with \"name\" key) needed to answer the query. You may include multiple tools.\n"
    "- Each tool must appear as: {{\"name\": \"exact_tool_name\"}}\n"
    "NEVER invent tool names—only use those listed above.\n"
    "NEVER include explanations, markdown, code blocks, extra fields, natural language, or formatting.\n"
    "Your response MUST be a valid JSON array. Each object MUST contain exactly one field: \"name\" (string), with the exact tool name.\n"
    "DO NOT include any other keys (such as \"parameters\", \"functions\") unless explicitly required by the system.\n"
    "Return ONLY the JSON array and nothing else.\n"
    'Example: [{{"name": "multi_search"}}, {{"name": "deep_reason"}}]'
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

