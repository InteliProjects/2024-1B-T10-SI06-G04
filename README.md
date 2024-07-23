# Inteli - Instituto de Tecnologia e Liderança

<p align="center">
<a href= "https://www.inteli.edu.br/"><img src="./assets/inteli.png" alt="Inteli - Instituto de Tecnologia e Liderança" border="0"></a>
</p>

# Emotion - Processamennto de Linguagem Natural
## Foster
## Integrantes:
- <a href="https://www.linkedin.com/in/joao-pedro-brandao/">João Pedro de Moura</a>
- <a href="https://www.linkedin.com/in/lucas-galv%C3%A3o/">Lucas Galvão</a>
- <a href="https://www.linkedin.com/in/luizarsantana/">Luiza Santana</a>
- <a href="https://www.linkedin.com/in/pedro-faria-santos-10b4061b7/">Pedro Faria</a>
- <a href="https://br.linkedin.com/in/ricardo-novaes-24276b271 ">Ricardo Baumgart</a>
- <a href="https://www.linkedin.com/in/sophianobrega/">Sophia Nóbrega</a>
- <a href="https://www.linkedin.com/in/yan-m-coutinho/">Yan Coutinho</a>

## Professores:
### Orientador
- Renato Penha
### Instrutores
- Ana Cristina
- Fabiana Martins de Oliveira
- Pedro Teberga
- Ricardo José Missori
- Victor Hayashi

## :memo: Descrição
Neste projeto, desenvolvemos um sistema de análise de sentimentos para a Uber, com o objetivo de classificar os comentários dos usuários como positivos, negativos ou neutros. A principal meta é fornecer insights valiosos para a melhoria contínua dos serviços da empresa, identificando pontos fortes e áreas que precisam de atenção.

Inicialmente, recebemos da Uber um banco de dados contendo aproximadamente 3200 linhas de comentários de usuários. Em seguida, realizamos o pré-processamento dos dados para limpar e preparar os textos para a análise. Esta etapa envolveu a remoção de palavras irrelevantes (stopwords), lematização (redução das palavras à sua forma básica), stemming (redução das palavras aos seus radicais) e normalização dos textos.

Após o pré-processamento, fizemos uma análise exploratória dos dados para entender melhor a distribuição dos sentimentos, identificar padrões e detectar possíveis outliers. Utilizamos gráficos e estatísticas descritivas para facilitar essa compreensão.

Para a modelagem, utilizamos diferentes técnicas de vetorização, como Bag of Words (BoW), TF-IDF e Word2Vec, e treinamos vários modelos de machine learning, incluindo Naive Bayes e Random Forest. Além disso, implementamos o TinyBERT, uma versão compacta do modelo BERT, conhecida por sua eficiência em tarefas de processamento de linguagem natural (PLN).

Os modelos foram avaliados utilizando métricas como acurácia, precisão, recall e F1-score para assegurar a robustez dos resultados. A validação cruzada foi aplicada para garantir que os modelos generalizassem bem em dados não vistos.

Desenvolvemos também uma API utilizando Flask, permitindo a integração do sistema de análise de sentimentos com outras aplicações da Uber. A API possui endpoints que permitem a classificação de novos comentários em tempo real. Rigorosos testes unitários e de integração foram realizados para garantir a performance e precisão da API em diferentes cenários.

Concluímos que o sistema de análise de sentimentos desenvolvido é uma ferramenta poderosa para a Uber entender melhor a percepção dos usuários sobre seus serviços. Com os insights obtidos, a empresa pode tomar decisões informadas para melhorar a satisfação do cliente e aprimorar suas operações. A combinação de modelos avançados de machine learning com uma API robusta garante a eficiência e escalabilidade da solução, pronta para ser utilizada em um ambiente de produção.

Recomendamos a implementação contínua de novos dados para re-treinamento dos modelos, assegurando que o sistema se mantenha atualizado com as mudanças nas opiniões dos usuários. Explorar técnicas mais avançadas de deep learning e incorporar feedback direto dos usuários pode proporcionar melhorias adicionais no desempenho do sistema.
## :memo: Link de demonstração
![Vídeo](./documentacao/outros/img/video.mp4)
<video controls src="./documentacao/outros/img/video.mp4" title="Title">Vídeo de demonstração</video>

## :file_folder: Estrutura de pastas
```sh
├───.vscode
├───documentacao
│   └───outros
│       └───img
└───src
    ├───api
    │   └───__pycache__
    ├───notebooks
    ├───data
    └───postman
```

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:
- <b>documentação</b>: aqui estão todos os documentos do projeto bem como documentos complementares, na pasta "outros".
- <b>src</b>: Todo o código fonte criado para o desenvolvimento do projeto de PLN.
- <b>README.md</b>: arquivo que serve como guia introdutório e explicação geral sobre o projeto e a aplicação (o mesmo arquivo que você está lendo agora).

## :computer: Configuração para desenvolvimento e execução do código

Aqui encontram-se todas as instruções necessárias para a instalação de todos os programas, bibliotecas e ferramentas imprescindíveis para a configuração do ambiente de desenvolvimento.
1. Clone o repositório em questão.
2. No modo administrador, abra o "prompt de comando" ou o "terminal" e, após, abra a pasta "src/api" no diretório raiz do repositório clonado e digite o segundo comando:
```sh
npm install -r  requirements.txt
```
Isso instalará todas as dependências definidas no arquivo <b>requirements.txt</b> que são necessárias para rodar o projeto. Agora o projeto já está pronto para ser modificado. Caso ainda deseje iniciar a aplicação, digite o comando abaixo no terminal:
```sh
python slack_api.py
```
Esse comando iniciará a parte da aplicação responsável pelas mensagens enviadas no slack. Agora para rodar a api efetivamente deve ser executado o seguinte comando:
```sh
python api.py
```
5. Agora você pode acessar a aplicação através do link http://localhost:5000/
6. A api está rodando.

### :card_file_box: Histórico de lançamentos
* 0.5.0 - 20/06/2024
    * Realização do deploy do melhor modelo.
    * Código fonte em formato de script Python disponível no GitHub.
    * Registro de comentários sobre refinamentos realizados.
    * Integração do modelo em um serviço (API local, Gradio).
    * Descrição textual do serviço com detalhamento dos métodos e respostas esperadas.
    * Disponibilização de vídeo de demonstração no Google Drive.
    * Explicação do processo de deploy e escolha do modelo no vídeo.
    * Comunicação eficiente no vídeo e nos comentários do código.
    * Estrutura do código fonte para facilitar integração e entendimento.
    * Diagrama de implantação UML da solução.
    * Explicação textual alinhada com o diagrama de implantação.

* 0.4.0 - 07/06/2024
    * Implementação da API.
    * Notebook organizado com instalação, definição de funções e rotas da API.
    * Suporte à vetorização BoW, Word2Vec e outra além destas.
    * Suporte à classificação de sentimento usando Naive Bayes e outro classificador.
    * Exportação dos modelos utilizando pickle.
    * Rota da API para classificação de sentimento com a melhor combinação de modelo e vetorização.
    * Comunicação eficiente nas células de texto e código.
    * Documentação da API em Markdown no GitHub.
    * Explicações, linhas de raciocínio e justificativas para técnicas, processamento e algoritmos.
    * Comparação entre todos os modelos e justificativa da escolha do modelo final.

* 0.3.0 - 24/05/2024
    * Modelo utilizando Word2Vec.
    * Notebook com modelo Word2Vec utilizando Embedding Layer ou Naive Bayes.
    * Uso de vetores de palavras pré-treinados.
    * Processamento do corpus com etapas ilustradas em dataframes.
    * Descrição textual das métricas utilizadas.
    * Resultados dos modelos Word2Vec registrados no notebook.
    * Comunicação eficiente nas células de texto e código.
    * Documentação do modelo Word2Vec em Markdown no GitHub.
    * Explicações e justificativas para as técnicas e resultados.
    * Comparação entre modelos Word2Vec e BoW da sprint anterior.

* 0.2.0 - 10/05/2024
    * Modelo de Bag of Words (BoW).
    * Notebook organizado com instalação, testes isolados, definição de funções e processamentos/demonstrações.
    * Análise descritiva do corpus com gráficos e estatísticas.
    * Pré-processamento com figura ilustrativa do pipeline e exemplos.
    * Gerar arquivo CSV com o resultado do pré-processamento.
    * Descrição do modelo BoW e métricas associadas.
    * Resultados do modelo BoW registrados no notebook.
    * Comunicação eficiente nas células de texto e código.
    * Documentação do modelo BoW em Markdown no GitHub.
    * Explicações e justificativas para técnicas e resultados.

* 0.1.0 - 26/04/2024
    * Entendimento do Negócio.
    * Matriz de Avaliação de Valor Oceano Azul.
    * Matriz de Risco.
    * Canvas Proposta de Valor.
    * Análise financeira do projeto.
    * Entendimento da Experiência do Usuário.
    * Criação de Personas e User Stories.

## :globe_with_meridians: Tecnologias Utilizadas

1. **Python**:
   - Linguagem principal utilizada para o desenvolvimento do projeto.

2. **Jupyter Notebook**:
   - Utilizado para o desenvolvimento interativo e análise de dados (`.ipynb` files).

3. **Flask**:
   - Framework utilizado para a criação da API (`api.py`).

4. **Pandas**:
   - Biblioteca para manipulação e análise de dados (mencionada nos requisitos de análises descritivas e processamentos).

5. **Scikit-learn**:
   - Biblioteca para aprendizado de máquina, usada para modelos como Naive Bayes (`naive_bayes.ipynb`), KMeans (`kmeans.ipynb`), entre outros.

6. **NLTK / SpaCy**:
   - Bibliotecas de processamento de linguagem natural, possivelmente utilizadas para tokenização e outras etapas de pré-processamento (`pre_processing.ipynb`).

7. **Word2Vec / Gensim**:
   - Utilizadas para a vetorização de texto (`w2v.ipynb`).

8. **Matplotlib / Seaborn**:
   - Bibliotecas para criação de gráficos e visualizações (`descriptive_analysis.ipynb`).

9. **Pickle**:
   - Utilizado para a exportação dos modelos treinados.

10. **TinyBERT**:
    - Implementação de modelos baseados em BERT para processamento de linguagem natural (`tiny_BERT.ipynb`).

11. **GitHub**:
    - Plataforma de versionamento de código e colaboração do projeto.

12. **Markdown**:
    - Utilizado para documentação do projeto (`README.md`).

13. **Postman**:
    - Ferramenta para testar APIs (`uber.postman_collection`).


## :clipboard: Licença/License
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Inteli-College/2024-1B-T10-SI06-G04">MODELO GIT INTELI</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.inteli.edu.br/">Inteli, <a href="https://github.com/joaopedrobrandao">João Pedro de Moura</a>, <a href="https://github.com/LucasG99">Lucas Galvão</a>, <a href="https://github.com/luizarsantana">Luiza Santana</a>, <a href="https://github.com/pedrofariasantos">Pedro Faria</a>, <a href="https://github.com/RicardoBMN">Ricardo Baumgart</a>, <a href="https://github.com/sophisnobrega">Sophia Nóbrega</a>, <a href="https://github.com/YanMCoutinho">Yan Coutinho</a> is licensed under <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
