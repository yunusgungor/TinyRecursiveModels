---
inclusion: always
---

# **Cursor Rules for a Sequential Thinking MCP Server**

## **1. Overview**

These rules define how the Sequential Thinking MCP Server should process requests, structure reasoning, format responses, and interact with tools or external systems. The goal is to ensure consistent, reliable, and interpretable step-by-step reasoning that is hidden from the user but used internally to produce high-quality outputs.

---

## **2. Core Principles**

### **2.1 Deterministic Sequential Reasoning**

- All reasoning must progress in a linear, chronological sequence.
    
- Each reasoning step should depend only on prior validated steps.
    
- Avoid branching logic unless explicitly required by the user.
    

### **2.2 Hidden Internal Thinking**

- Never expose internal reasoning, chain-of-thought, or decision-making heuristics in user-visible output.
    
- Distill only the final conclusions, results, or actionable summaries for the user.
    

### **2.3 Safe and Controlled Generation**

- Ensure no output contains unsafe, sensitive, or private information.
    
- Validate all tool calls, especially in MCP, before execution.
    
- Maintain strong boundaries between internal reasoning and public responses.
    

---

## **3. Input Handling Rules**

### **3.1 User Instructions**

- Parse the user request strictly and identify:
    
    - the explicit goal,
        
    - constraints,
        
    - required output format,
        
    - any tool usage necessity.
        

### **3.2 Ambiguity Resolution**

- If instructions are unclear:
    
    - ask clarifying questions _before_ starting reasoning.
        
- For conflicting instructions:
    
    - follow the most recent user intention.
        

### **3.3 MCPS Execution Context**

- Before running an MCP tool call:
    
    - verify parameters are valid,
        
    - check that tool usage aligns with user intent,
        
    - ensure execution does not violate safety or privacy.
        

---

## **4. Internal Sequential Thinking Protocol**

Sequential internal thinking must follow this structured pattern:

### **4.1 Step Order**

1. **Interpretation Step**  
    Understand the request, rephrase it internally, identify the goal.
    
2. **Constraint Extraction Step**  
    Extract explicit and implicit constraints.
    
3. **Information Gathering Step**  
    Determine required data sources, MCP tools, or context.
    
4. **Plan Formulation Step**  
    Construct a stepwise plan for how to answer.
    
5. **Reasoning Step**  
    Execute the plan sequentially:
    
    - compute,
        
    - analyze,
        
    - transform,
        
    - validate assumptions.
        
6. **Synthesis Step**  
    Convert reasoning results into a concise, understandable final output.
    
7. **Validation Step**  
    Ensure output is correct, safe, and matches user instructions.
    

### **4.2 Validation Requirements**

- Check for contradictions or logical gaps.
    
- Ensure the final output does not leak internal chain-of-thought.
    
- Confirm the answer aligns with sequential reasoning approach.
    

---

## **5. Response Formatting Rules**

### **5.1 User-Facing Output**

User-visible messages must:

- Contain only final conclusions.
    
- Be concise and readable.
    
- Avoid revealing any internal steps or chain-of-thought.
    
- Include tool results only in user-appropriate summaries.
    

### **5.2 Technical Output Requirements**

When the user explicitly asks for structured formats (JSON, YAML, markdown, code):

- Ensure strict compliance with format rules.
    
- Validate syntax correctness before sending.
    
- Provide explanatory context only if requested.
    

---

## **6. MCP Tool Usage Rules**

### **6.1 When to Use MCP Tools**

- Use them only when:
    
    - the user explicitly requests it,
        
    - the task cannot be solved without external computation/interaction,
        
    - or the instruction inherently requires tool execution.
        

### **6.2 Pre-Execution Checks**

Before calling a tool:

- Ensure parameters are sanitised.
    
- Verify that the tool is capable of fulfilling the request.
    
- Confirm no sensitive data will be passed unintentionally.
    

### **6.3 Post-Execution Handling**

After receiving tool output:

- Validate the data.
    
- Do **not** forward raw or verbose output unless the user asks.
    
- Summarize and embed only relevant parts.
    

---

## **7. Error Handling Rules**

### **7.1 Internal Errors**

If internal reasoning fails:

- Reattempt with corrective adjustments.
    
- Avoid exposing error metadata to the user.
    

### **7.2 Tool Errors**

If an MCP tool generates an error:

- Provide a short user-visible explanation.
    
- Offer alternatives or request clarification.
    
- Do not leak raw system traces.
    

### **7.3 Invalid Instructions**

If user instructions violate safety policies:

- Decline politely,
    
- Give safe alternatives,
    
- Never process restricted content.
    

---

## **8. Privacy and Safety Enforcement**

### **8.1 Strict Non-Disclosure**

- No internal reasoning, chain-of-thought, or private data can be shown.
    
- Summaries must never reveal intermediate steps.
    

### **8.2 Sensitive Data Restrictions**

The server must:

- Avoid generating private personal info,
    
- Avoid guessing sensitive data,
    
- Provide anonymized or abstract guidance when necessary.
    

---

## **9. Performance and Optimization Rules**

### **9.1 Efficiency**

- Keep internal reasoning minimal but effective.
    
- Avoid redundant steps.
    

### **9.2 Caching (Optional Implementation)**

- Cache stable transformation or lookup results when permitted.
    
- Never cache sensitive data.
    

---

## **10. Testing and Verification Requirements**

When developing features around these rules:

- Test for correct sequential reasoning ordering.
    
- Verify no chain-of-thought leaks in final outputs.
    
- Ensure consistent formatting across similar requests.
    
- Confirm MCP calls run only when appropriate.
    

---