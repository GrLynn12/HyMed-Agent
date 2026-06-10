---
name: disease_treatment
description: 处理疾病治疗方式、治疗方案和“怎么办”类问题
preferred_tools: [medical_graph_search, medical_vector_search]
required_memory: []
---

1. 先使用知识图谱查询标准治疗方法。
2. 如果图谱只返回“药物治疗”“手术治疗”等泛化词，证据不充分，继续使用向量检索。
3. 不得根据疾病名称自行生成具体处方药、剂量或疗程。
4. 疾病类型、部位或严重程度不明确时，回答中说明这一限制。
5. 最终结论必须能映射到工具返回的证据。
