# 1、先在shell终端执行这个命令，构建judge_modge_args参数，主要需要保证DEEPSEEK_API_KEY有值
export JUDGE_MODEL_ARGS="$(python - <<'PY'
import os, json

args = {
      "model_id": "deepseek-v4-pro",
      "api_url": "https://api.deepseek.com",
      "api_key": os.environ["DEEPSEEK_API_KEY"],
      "generation_config": {
          "temperature": 0.0,
          "max_tokens": 1024
      },
      "score_type": "pattern",
      "score_pattern": "(A|B)",
      "score_mapping": {
          "A": 1.0,
          "B": 0.0
      },
      "prompt_template": """你是一个严格的中文关键词抽取评测裁判。不要输出分析过程。

  请判断模型抽取的关键词是否与参考关键词基本等价。最终只返回 A 或 B。

  判定规则：
  1. 关键词顺序不同不扣分。
  2. 分隔符不同不扣分。
  3. 同义表达、合理简称可以接受。
  4. 遗漏核心主题关键词，判为 B。
  5. 加入明显无关关键词，判为 B。
  6. 输出整段句子而不是关键词列表，判为 B。
  7. 如果模型把中文关键词翻译成英文，除非语义完全准确且覆盖核心主题，否则判为 B。

  A: 合格
  B: 不合格

  [题目]
  {question}

  [参考关键词]
  {gold}

  [模型输出]
  {pred}

  最终答案只返回 A 或 B。"""
  }

print(json.dumps(args, ensure_ascii=False))
PY
)"

# 2、然后再执行这个参数，完成评估
evalscope eval \
    --model Qwen3-14B \
    --api-url "http://127.0.0.1:8000/v1" \
    --datasets general_qa \
    --dataset-args '{"general_qa":{"local_path":"benchmark/custom_eval/text/qa","subset_list":["keyword_extraction"],"prompt_template":"{query}"}}' \
    --judge-strategy llm \
    --judge-model-args "$JUDGE_MODEL_ARGS" \
    --generation-config '{"temperature":0.0,"max_tokens":256}' \
    --eval-batch-size 4