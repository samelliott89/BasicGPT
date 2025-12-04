# # DON'T: Ask Claude to generate a Mamba implementation
# # (I might get it subtly wrong)

# # DO: Extract the actual Mamba code from the repo
# # 1. Use Real Code as Ground Truth (not synthetic)

# def mine_real_implementations():
#     """
#     The ground truth comes from HUMAN-WRITTEN code.
#     Claude just does mechanical extraction/transformation.
#     """
#     real_code = extract_from_repo("state-spaces/mamba")  # Human ceiling, not Claude ceiling
    
#     # Claude only generates the PROMPT, not the solution
#     prompt = generate_prompt_from_code(real_code)  # Safe: just describing what exists
    
#     return {
#         "prompt": prompt,           # Claude-generated (low risk)
#         "solution": real_code,      # Human-written (no ceiling)
#     }

# # The reward signal comes from REALITY, not from Claude's judgment

# def get_reward(generated_code):
#     # Claude's understanding is IRRELEVANT here
#     # The universe (Python interpreter) is the judge
    
#     parses = try_parse(generated_code)
#     runs = try_execute(generated_code)
#     trains = try_train_steps(generated_code, n=100)
    
#     # Ground truth from execution, not from model evaluation

# ### 3. Human Expert Annotation for the Hard Stuff

# # For the cutting-edge challenges where no model understands well enough:
# # ```
# # Source                          Who creates what
# # ─────────────────────────────────────────────────────────
# # Standard patterns (80%)    →    Synthetic generation OK
# # Recent papers (15%)        →    Mine from author repos
# # Novel/tricky cases (5%)    →    Human expert writes