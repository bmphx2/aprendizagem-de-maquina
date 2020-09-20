# Trabalho Final da Disciplina de Aprendizagem De Máquina

Alunos:

- Bruno Gonçalves de Oliveira (bruno.mphx2@gmail.com)
- Diogo Cezar Teixeira Batista (diogocezar@ufpr.br)

Será necessário executar alguns imports:

```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## TODO

- Ferramenta para obtenção dos dados https://github.com/gavinr/github-csv-tools

- a base de teste é do Wildfly, os de treinamento sao dos couch base, jgit, e okhttp

- list of words, veio de uma checagem de palavras que mais apareciam nos textos dos commits que eram de segurança...

```
most_freq = heapq.nlargest(40, wordfreq, key=wordfreq.get)
		print(most_freq)
```
