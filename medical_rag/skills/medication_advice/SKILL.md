---
name: medication_advice
description: 处理药品选择、外用药、禁忌、副作用、剂量和药物冲突
preferred_tools: [medical_graph_search, medical_vector_search]
required_memory: [allergy, medication]
---

1. 回答前必须检查相关过敏史和当前用药记忆。
2. 具体药物名称必须有检索证据支持；只有“药物治疗”时不得自行补药名。
3. 没有剂量证据时禁止生成具体剂量、频次和疗程。
4. 发现潜在过敏或用药冲突时，不直接推荐该药，并建议医生或药师确认。
5. 无法确定疾病类型或适应证时，给出证据边界而不是猜测。
