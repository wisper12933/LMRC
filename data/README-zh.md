# 数据说明
## 关系候选生成 (Relation Candidate Proposal)
训练与测试使用原 DocRED 与 Re-DocRED 数据集

## 关系分类 (Relation Classification)
由 DocRED 与 Re-DocRED 数据集处理得到，用于微调/测试大语言模型 (LLM) 的数据集

### 训练集 (train)
- `step2` 后缀文件：用于微调 LLM 进行关系分类的训练数据  
- `baseline` 后缀文件：用于微调LLM进行完整的文档级关系抽取的训练数据

*注：测试 LMRC 需使用 `Relation Candidate Proposal` 中的测试集与开发集*