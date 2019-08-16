# A reference implementation of algorithms for distributions over spanning trees

This project provides a reference implementation of the inference algorithms for
globally normalized distributions over directed spanning trees.  In natural
language processing, such models are used to develop nonprojective dependency
parsers.  The current implementation shows the matrix-tree theorem in
action. The implementation is carefully tested against brute-force algorithms
that explicitly marginalizes over directed trees.


**Citation:** If you found this useful, please cite it as
```bibtex
@software{vieira-spanningtree,
  author = {Tim Vieira},
  title = {A reference implementation of algorithms for distributions over spanning trees},
  url = {https://github.com/timvieira/spanning_tree}
}
```

**References**: In 2007, three different groups published similar methods for
inference in discriminative nonprojective dependency parsing.

 - Koo, Globerson, Carreras and Collins (EMNLP'07)
   [Structured Prediction Models via the Matrix-Tree Theorem](https://www.aclweb.org/anthology/D07-1015)

 - Smith and Smith (EMNLP'07)
   [Probabilistic Models of Nonprojective Dependency Trees](https://www.aclweb.org/anthology/D07-1014)

 - McDonald & Satta (IWPT'07)
   [On the Complexity of Non-Projective Data-Driven Dependency Parsing](https://www.aclweb.org/anthology/W07-2216)
