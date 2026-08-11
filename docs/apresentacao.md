# RESUMO

Esta apostila documenta a construção completa de um modelo de linguagem do tipo
*Transformer*, escrito integralmente do zero em Python, sem o uso de bibliotecas de alto
nível para as partes conceituais. O percurso vai da contagem de pares de caracteres em um
modelo bigrama até um modelo multimodal capaz de gerar imagens, passando por diferenciação
automática, mecanismo de atenção, tokenização por *Byte Pair Encoding*, otimização,
treinamento em precisão reduzida, paralelismo de dados, curadoria de *corpus*, cache de
inferência, quantização, ajuste fino supervisionado, aprendizado por reforço e servimento
em produção.

O texto adota uma restrição metodológica explícita: **nenhuma afirmação quantitativa é
publicada sem medição**. Toda tabela de resultados provém da execução dos programas que
acompanham o material, e os casos em que a medição contrariou a hipótese inicial do autor
foram mantidos no texto, documentados como tal. Essa escolha responde a um objetivo
pedagógico que se considera tão importante quanto o conteúdo técnico: demonstrar que uma
explicação plausível e uma explicação correta são coisas distintas, e que apenas o
experimento as separa.

A organização segue o programa do curso **LLM101n**, de Andrej Karpathy e da Eureka Labs,
distribuído em dezessete capítulos. A exposição é feita em português; a terminologia
técnica e os comentários de código permanecem em inglês, por serem a forma em que o leitor
os encontrará na literatura e nas bibliotecas.

**Palavras-chave:** Modelos de linguagem. Redes neurais. *Transformer*. Processamento de
linguagem natural. Aprendizado profundo. Metodologia experimental.

---

# APRESENTAÇÃO

## A quem se destina

O material pressupõe **Python básico**, noção elementar de programação e matemática de
nível universitário inicial — derivadas parciais e multiplicação de matrizes. Não
pressupõe qualquer contato prévio com aprendizado de máquina. Todo conceito da área é
introduzido no ponto em que se torna necessário, e nunca antes.

## Como o material está organizado

Cada capítulo é uma unidade fechada com quatro componentes:

| Componente | Função |
|---|---|
| Texto expositivo | a teoria e os resultados medidos |
| Programas | a implementação, do zero, comentada |
| Exercícios | ao final de cada capítulo |
| Gabaritos | reunidos no Apêndice A, deliberadamente separados |

A separação dos gabaritos é intencional. Um exercício cuja resposta está na página seguinte
não é um exercício; nesta edição as soluções ocupam um apêndice, o que exige do leitor uma
decisão consciente de consultá-las.

Os capítulos são cumulativos. O autógrado escrito no Capítulo 2 é o mesmo usado no Capítulo
5; o *Transformer* do Capítulo 5 é o mesmo que, no Capítulo 17, recebe imagens sem uma
única linha alterada. Essa continuidade é o argumento central do curso, e ler os capítulos
fora de ordem a desfaz.

## Sobre os números apresentados

Todas as medições foram obtidas em uma única máquina, sem GPU dedicada, e devem ser lidas
como **ordens de grandeza e relações entre condições**, não como marcas absolutas. O que
importa nas tabelas é a comparação entre as linhas — o ganho de um cache, o custo de uma
quantização, a diferença entre duas taxas de aprendizado — e essa comparação é robusta
mesmo quando o valor absoluto não é reprodutível em outro equipamento.

Duas advertências, ambas resultado de erros cometidos durante a redação e documentados nos
capítulos onde ocorreram:

1. **Conclusões dependem do orçamento de treino.** Diversos resultados deste material se
   inverteram ao mudar o número de passos de otimização. Um experimento curto pode não
   apenas subestimar um efeito, mas apontá-lo na direção oposta.

2. **Uma métrica que melhora não é um sistema que melhorou.** Vários capítulos exibem casos
   em que o número escolhido para medir o sucesso subiu enquanto o comportamento de
   interesse piorava.

## Reprodutibilidade

O repositório inclui um verificador (`tools/smoke_test.py`) que executa todos os programas
do curso e reporta o estado de cada um. A última execução registrada percorreu os
dezessete capítulos e setenta e oito programas sem falhas.
