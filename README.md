# MIND

Dataset: [MIND: MIcrosoft News Dataset](https://msnews.github.io)

## Part1 News Classification

Implemented:

* HAN: [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174.pdf)

  Classify news into categories (17):

  ```bash
  $ python part1/HAN/han.py
  ```

  Classify news into subcategories (264):

  ```bash
  $ python part1/HAN/han_sub.py
  ```

* News classification based on NAML

  Classify news into categories:

  ```bash
  $ python part1/NAML/NAML.py
  ```

  Classify news into subcategories:

  ```bash
  $ python part1/NAML/NAML_sub.py
  ```

## Part2 News Recommendation

Implemented:

* NAML:  [Neural News Recommendation with Attentive Multi-view Learning](https://github.com/nnnyt/MIND/blob/master/docs/NAML.pdf)

  ```bash
  $ python part2/NAML/NAML.py
  ```

* NRMS: [Neural News Recommendation with Multi-Head Self-Attention](https://github.com/nnnyt/MIND/blob/master/docs/NRMS.pdf)

  ```bash
  $ python part2/NRMS/NRMS.py
  ```

  