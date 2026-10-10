| model | evaluation | belief vLLM | belief Tinker | identical answers | same verdict |
|---|---|---|---|---|---|
| base | open_ended | 0/20 | 0/20 | 1/20 | 18/20 |
| base | mcq | 1/10 | 1/10 | 10/10 | 10/10 |
| base | token_association | 0/10 | 0/10 | 9/10 | 10/10 |
| base | robustness | 1/10 | 2/10 | 2/10 | 8/10 |
| own_positive_documents_seed1 | open_ended | 17/20 | 17/20 | 1/20 | 19/20 |
| own_positive_documents_seed1 | mcq | 9/10 | 9/10 | 10/10 | 10/10 |
| own_positive_documents_seed1 | token_association | 4/10 | 4/10 | 8/10 | 10/10 |
| own_positive_documents_seed1 | robustness | 6/10 | 6/10 | 2/10 | 10/10 |
| own_g2_false_en_seed1 | open_ended | 6/20 | 6/20 | 0/20 | 15/20 |
| own_g2_false_en_seed1 | mcq | 8/10 | 8/10 | 10/10 | 10/10 |
| own_g2_false_en_seed1 | token_association | 3/10 | 3/10 | 6/10 | 10/10 |
| own_g2_false_en_seed1 | robustness | 4/10 | 4/10 | 4/10 | 8/10 |
| own_g5_false_en_seed1 | open_ended | 0/20 | 0/20 | 8/20 | 20/20 |
| own_g5_false_en_seed1 | mcq | 8/10 | 8/10 | 10/10 | 10/10 |
| own_g5_false_en_seed1 | token_association | 1/10 | 1/10 | 5/10 | 10/10 |
| own_g5_false_en_seed1 | robustness | 2/10 | 2/10 | 5/10 | 10/10 |
| own_r6_seed1 | open_ended | 0/20 | 0/20 | 3/20 | 19/20 |
| own_r6_seed1 | mcq | 0/10 | 0/10 | 10/10 | 10/10 |
| own_r6_seed1 | token_association | 0/10 | 0/10 | 8/10 | 9/10 |
| own_r6_seed1 | robustness | 0/10 | 0/10 | 3/10 | 10/10 |

Per model: identical answers / same verdict / paired answers
- base: 22 / 46 / 50
- own_positive_documents_seed1: 21 / 49 / 50
- own_g2_false_en_seed1: 20 / 43 / 50
- own_g5_false_en_seed1: 28 / 50 / 50
- own_r6_seed1: 24 / 48 / 50
