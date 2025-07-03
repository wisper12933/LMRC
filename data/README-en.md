# Data Description  
## Relation Candidate Proposal  
The original DocRED and Re-DocRED datasets are used for training and testing the smaller scaled model in RCP stage.  

## Relation Classification  
Processed datasets derived from DocRED and Re-DocRED, used for fine-tuning/testing LLMs in RC stage.  

### Training Data (train)  
- Files with `step2` suffix: Training data for fine-tuning LLMs on relation classification.  
- Files with `baseline` suffix: Training data for fine-tuning LLMs on full document-level relation extraction.  

*Note: Use the test/dev sets from "Relation Candidate Proposal" for testing LMRC.*  