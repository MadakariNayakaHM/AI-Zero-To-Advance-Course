# Introduction to Generative AI

> IIIT Dharwad · Semester 1

## Notes

| # | Topic | Slides |
|:-:|:--|:--|
| 01 | [Foundations: AI, ML, DL & the Learning Setup](01-AI-ML-DL-Foundations.md) | 1–17 |
| 02 | [Neural Networks: Why, What & How They Compute](02-Neural-Networks-Fundamentals.md) | 18–26 |

**Notebook:** [neural networks from scratch](code/neural_networks_from_scratch.ipynb) covers the forward pass by hand, XOR, a network trained with hand-written backpropagation (NumPy only), a gradient check, softmax temperature and parameter counting.

**Coming up:** training & backpropagation · optimisers · sequence models · attention · Transformers & LLMs · VAEs, GANs & diffusion

## Notation on the slides

- **Slide 23:** the text says "`Wⱼᵢ` connects input `j` to neuron `i`" but the formula uses `Wᵢⱼxⱼ`. The notes read `Wᵢⱼ` as *from input j to neuron i*, so `a = Wx`.
- **Slide 26:** the output weights are written as `n × k`. Under the column-vector convention `a = Wh` they are `k × n`.

## Related notes

- Paradigms, and how ChatGPT is trained → [ML Paradigms 01](../../Machine-Learning-Paradigms/Notes/01-Introduction-to-ML-Paradigms.md)
- Loss functions and gradient descent → [ML Paradigms 03](../../Machine-Learning-Paradigms/Notes/03-Supervised-Learning-Regression.md)
- `Wx` as a linear combination, rank, feature space → [Applied Mathematics](../../Applied-Math-For-AI-DS/Notes/README.md)
