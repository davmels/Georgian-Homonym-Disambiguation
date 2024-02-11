# Georgian-homonym-disambiguation
This repository contains all the training and testing datasets for the Georgian homonym disambiguation task. Additionally, it includes the code for model creation, training and testing.
## Dataset
At this point I've considered only the homonym: "ბარი" and it's different grammatical forms obtaining 7522 sentences.

The dataset includes:

- 763 sentences using "ბარი" as a "shovel" labaled with 0
- 1846 sentences using "ბარი" as a "lowland" labeld with 1
- 3320 sentences using "ბარი" as a "cafe" labeled with 2 
- 1593 sentences where the homonym is used in a different context, labeled with 3 (Although these sentences could be further classified by the definitions of the homonyms, for this project I've ignored other usages).

## Models
- Transformers &mdash; Fill-Mask
- Transformers &mdash; Text Classification
- Recurrent neural networks &mdash; LSTM

I've also incorporated the Word2Vec model trained on the CC100 Georgian dataset, utilizing it for word vectorization in my recurrent neural network.
