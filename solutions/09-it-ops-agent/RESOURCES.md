# Solution 09 — Learning resources

## Agents and tool calling

* Hugging Face Agents Course (free; tools, ReAct loop, frameworks): <https://huggingface.co/learn/agents-course>
* Anthropic, "Building effective agents" (patterns: workflows vs agents, when to use which): <https://www.anthropic.com/engineering/building-effective-agents>
* OpenAI-compatible function calling format (used by llama-server): llama.cpp function-calling docs
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>
* Granite 4.2 tool calling (model card, chat template): <https://huggingface.co/ibm-granite/granite-4.2-8b>

## Agent frameworks (next steps beyond the plain loop)

* LangGraph (ppc64le wheel on IBM's index): <https://langchain-ai.github.io/langgraph/>
* BeeAI Framework (IBM-led, open source): <https://github.com/i-am-bee/beeai-framework>
* Model Context Protocol (expose these tools to any MCP client): <https://modelcontextprotocol.io/>

## Safety for agents that touch infrastructure

* OWASP Top 10 for LLM Applications (prompt injection, excessive agency): <https://genai.owasp.org/llm-top-10/>
* Red Hat: sudoers and least privilege on RHEL 9: <https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/security_hardening/>

## Linux and IBM Power operations

* RHEL 9 system administration docs: <https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9>
* IBM Power performance: `lparstat`, SMT, entitlement (IBM Docs, "Performance tools for Linux on Power"): <https://www.ibm.com/docs/en/linux-on-systems>
* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheels: <https://wheels.developerfirst.ibm.com/ppc64le/linux> · RocketCE: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a tool `certificate_expiry(host, port)` and a scenario that uses it.
2. Rewrite `src/agent.py` as a LangGraph graph with an explicit approval node; compare readability.
3. Expose `tools_ops.py` as an MCP server so other assistants can use the same tools.
4. Feed Prometheus/Alertmanager alerts into `/tasks` and have the agent pre-diagnose each alert.
