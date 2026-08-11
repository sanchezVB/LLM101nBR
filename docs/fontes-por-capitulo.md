# APÊNDICE B — FONTES PRIMÁRIAS POR CAPÍTULO

Este apêndice associa cada capítulo aos trabalhos em que as técnicas apresentadas foram
originalmente propostas. Serve a dois propósitos: permitir ao leitor ir à fonte quando a
exposição didática não bastar, e deixar explícito que nada neste material é invenção do
autor — a contribuição está na ordem, na implementação e na medição, não nos métodos.

| Cap. | Tema | Fontes primárias |
|---|---|---|
| 1 | Modelo bigrama | SHANNON (1948); JURAFSKY; MARTIN (2024) |
| 2 | Diferenciação automática | KARPATHY (2023) |
| 3 | *N*-grama neural | BENGIO *et al.* (2003) |
| 4 | Atenção | BAHDANAU; CHO; BENGIO (2015); VASWANI *et al.* (2017) |
| 5 | *Transformer* | VASWANI *et al.* (2017); RADFORD *et al.* (2018, 2019); BA; KIROS; HINTON (2016); XIONG *et al.* (2020) |
| 6 | Tokenização (BPE) | GAGE (1994); SENNRICH; HADDOW; BIRCH (2016) |
| 7 | Otimização | KINGMA; BA (2015); LOSHCHILOV; HUTTER (2019); GLOROT; BENGIO (2010); HE *et al.* (2015) |
| 8 | Dispositivo e desempenho | PASZKE *et al.* (2019) |
| 9 | Precisão reduzida | MICIKEVICIUS *et al.* (2018) |
| 10 | Treinamento distribuído | LI *et al.* (2020); RAJBHANDARI *et al.* (2020) |
| 11 | Dados e escala | KAPLAN *et al.* (2020); HOFFMANN *et al.* (2022); ASSIS (1881, 1891, 1899) |
| 12 | Inferência e *KV-cache* | SHAZEER (2019); POPE *et al.* (2023) |
| 13 | Quantização | JACOB *et al.* (2018); DETTMERS *et al.* (2022); FRANTAR *et al.* (2023) |
| 14 | Ajuste fino supervisionado | WEI *et al.* (2022); OUYANG *et al.* (2022) |
| 15 | Aprendizado por reforço | WILLIAMS (1992); SCHULMAN *et al.* (2017); STIENNON *et al.* (2020); OUYANG *et al.* (2022); DEEPSEEK-AI (2025) |
| 16 | Servimento e produção | YU *et al.* (2022); KWON *et al.* (2023) |
| 17 | Multimodalidade | OORD; VINYALS; KAVUKCUOGLU (2017); BENGIO; LÉONARD; COURVILLE (2013); RAMESH *et al.* (2021); ESSER; ROMBACH; OMMER (2021); DOSOVITSKIY *et al.* (2021); HO; JAIN; ABBEEL (2020); LECUN; CORTES; BURGES (1998) |

## Sobre o programa do curso

A sequência dos dezessete capítulos segue o programa do **LLM101n**, de Andrej Karpathy e
da Eureka Labs (KARPATHY, 2024). A escolha dos tópicos e sua ordem são daquele programa; a
redação, a implementação, os experimentos e todos os números apresentados são deste
material.

## Sobre os dados

| Uso | Origem |
|---|---|
| Nomes próprios (cap. 1 e 3) | IBGE, Censo Demográfico 2010 |
| *Corpus* literário (cap. 6, 11-16) | Machado de Assis, via Project Gutenberg, domínio público |
| Dígitos manuscritos (cap. 17) | MNIST (LECUN; CORTES; BURGES, 1998) |

Todas as três fontes são de uso livre. A escolha não foi apenas jurídica: um material
didático cujos dados o leitor não pode baixar e reexecutar não é reprodutível, e a
reprodutibilidade é uma premissa declarada desta apostila.
