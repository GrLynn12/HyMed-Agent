---
name: prognosis_and_risk
description: 处理不治疗后果、自愈、严重程度、复发、风险和预后
preferred_tools: [medical_vector_search]
required_memory: []
---

1. 优先使用向量知识库检索相似医疗问答和长文本证据。
2. 区分常见后果、可能风险和需要及时就医的警示情况。
3. 不把“可能”表述为必然，不夸大低概率严重结果。
4. 若检索结果与问题主题不一致，应重新组织查询，而不是直接回答。
5. 最终回答应明确证据能支持到什么程度。
