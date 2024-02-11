# Georgian-homonym-disambiguation
This repository contains all the training and testing datasets for the Georgian homonym disambiguation task. Additionally, it includes the code for model creation, training and testing.
## Dataset
At this point I've considered only the homonym: "ბარი" and it's different grammatical forms obtaining 7522 sentences.

The dataset includes:

- 763 sentences using "ბარი" as a "shovel"
- 1846 sentences using "ბარი" as a "lowland"
- 3320 sentences using "ბარი" as a "cafe"
- 1593 sentences where the homonym is used in a different context

## Models
- Transformers &mdash; Fill-Mask
- Transformers &mdash; Text Classification
- Recurrent Neural Networks &mdash; LSTM
