# BRISC 2025 — pontos para discutir com os coautores

6 de outubro de 2026 · Preparação da revisão do manuscrito e da resposta ao revisor 1

## 1. O que está confirmado e o que continua por resolver

Os resultados principais foram reproduzidos na partição histórica do estudo. O problema encontrado não é a inexistência desses resultados: é a mistura de execuções em alguns elementos estatísticos e a interpretação demasiado forte da curadoria pHash. Há exclusões sustentadas por duplicação exata, mas também pares visivelmente distintos que passaram o filtro.

Foi repetida nesta data a verificação do corpus original e a pesquisa pHash em segundo plano. As 6000 imagens correspondem ao manifesto, sem ficheiros em falta, adicionais ou com conteúdo alterado. A pesquisa voltou a produzir 1349 pares, e a comparação dos pixels confirmou os valores abaixo. Não houve remoções, reposições de imagens, novos treinos ou alteração da partição. A revisão humana completa das exclusões continua pendente.

Este documento propõe correções para discussão. Não significa que tenham sido aplicadas no Prism, no manuscrito ou na resposta ao revisor. Os resultados dos modelos apresentados aqui vêm da auditoria de inferência anterior; não foram obtidos repetindo os treinos nesta execução pHash.

## 2. Parágrafo para enviar aos professores

Na conferência dos dados e scripts com o manuscrito e a resposta ao revisor 1, identificámos um problema na justificação do pHash: há pares visualmente distintos entre as exclusões, incluindo um com distância 2. Das 909 exclusões pHash, 83 correspondem a pares com pixels RGB idênticos; as restantes 826 precisam de outra avaliação, não sendo necessariamente exclusões erradas. Sugiro retirar a garantia de que o limiar é conservador e identifica apenas imagens indistinguíveis, distinguir duplicação exata de exclusão por semelhança e decidir que validação adicional apresentar ao revisor. As accuracies principais foram reproduzidas na partição histórica, mas desconhecemos o efeito de uma curadoria revista. Há também um IC da CNN associado à execução anterior, comparações RadImageNet desatualizadas e uma diferença entre o resultado clássico original e o reajuste Python. Podemos corrigir esses elementos com a evidência já disponível; a validade da curadoria exige uma decisão separada.

## 3. Resultados da auditoria pHash repetida

| Verificação | Resultado |
|---|---:|
| Imagens originais verificadas por SHA256 | 6000 |
| Partição histórica retida | 5041: 4363 treino / 678 teste |
| Exclusões históricas por MD5 / pHash | 50 / 909 |
| Pares candidatos no corpus original, distância ≤5 | 1349 |
| Imagens distintas envolvidas nos candidatos | 1834 |
| Pares candidatos entre classes diferentes | 23 |
| Pares candidatos entre treino e teste | 440 |
| Pares candidatos entre classes e splits diferentes | 3 |
| Pares com ficheiros idênticos por SHA256 | 55 |
| Pares com dimensões e pixels RGB idênticos | 159 |
| Pares RGB idênticos entre treino e teste | 110 |
| Pares RGB idênticos entre classes diferentes | 0 |
| Pares históricos pHash recuperados | 909 de 909 |
| Pares históricos pHash com pixels RGB idênticos | 83 |

Pares não são imagens removidas: uma imagem pode aparecer em vários pares. Os 159 pares RGB idênticos formam 134 grupos, com 278 imagens. Manter uma imagem por grupo retiraria 144 cópias redundantes. Este é um cenário de referência para cópias exatas, não uma proposta de benchmark final: podem continuar a existir versões transformadas da mesma imagem ou cortes relacionados.

Entre as 909 exclusões pHash históricas, 296 pares ligam treino e teste e 613 são do mesmo split. As exclusões pHash retiraram 599 imagens de treino e 310 de teste. Há 10 pares históricos com classes diferentes. Portanto, 909 exclusões por semelhança não equivale a 909 casos de leakage treino–teste.

A igualdade RGB foi calculada após leitura Pillow, sem redimensionamento ou alinhamento. Pixels diferentes não provam que duas imagens sejam independentes: podem ser cópias recomprimidas, transformações ou cortes próximos. Pixels idênticos confirmam igualdade da imagem descodificada, mas não dizem quantos pacientes existem.

### Sensibilidade do número de pares ao limiar

| Distância máxima | Pares | Entre classes | Entre treino/teste | RGB idênticos |
|---|---:|---:|---:|---:|
| 0 | 205 | 0 | 128 | 159 |
| 2 | 500 | 2 | 213 | 159 |
| 4 | 1349 | 23 | 440 | 159 |
| 5 | 1349 | 23 | 440 | 159 |

Esta é uma análise da contagem de candidatos, não da sensibilidade da accuracy. Não foi criado um novo dataset para cada limiar. Distâncias 4 e 5 dão os mesmos pares neste corpus; isso não demonstra que o limiar 4 seja válido. Mesmo a distância 0 inclui 46 pares sem igualdade RGB, que necessitam de outra avaliação.

A inspeção visual diagnóstica anterior cobriu os 23 candidatos entre classes e 18 exemplos da mesma classe, distribuídos pelas distâncias 0, 2 e 4. A seleção destes 18 exemplos foi determinística, não uma amostra aleatória para estimar a taxa de falsos positivos. Encontraram-se diferenças evidentes de padrões internos e orientação, incluindo pares axial/sagital. Não foi feita validação clínica dos rótulos nem adjudicação humana completa. Os campos de decisão, revisor e justificação permanecem em branco no CSV publicado.

## 4. Resultados dos modelos e correções estatísticas

| Configuração | Acertos / 678 | Accuracy | IC95 estratificado |
|---|---:|---:|---|
| CNN principal, três blocos | 636 | 93,81% | 91,89–95,43% |
| CNN anterior, três blocos | 631 | 93,07% | 91,00–94,84% |
| VGG16 softmax | 664 | 97,94% | 96,76–98,82% |
| VGG16 + LightGBM | 669 | 98,67% | 97,79–99,41% |
| RadImageNet ResNet50 | 489 | 72,12% | 69,17–75,37% |
| Clássico, novo reajuste Python | 614 | 90,56% | 88,50–92,63% |

Os IC apresentados usam 1000 resamples estratificados por classe, seed 42 e ordenação por classe/filename. São condicionais ao modelo treinado e à amostra observada. Não incluem variabilidade entre sementes de treino, agrupamento por paciente ou incerteza introduzida pela escolha de classificadores no teste.

O manuscrito associa a CNN de 93,81% ao IC de uma execução anterior. A accuracy, o F1 e a AUC da CNN principal confirmam-se; o IC deve ser substituído nas tabelas 1 e A1. A execução antiga deve continuar identificada separadamente.

O resultado clássico original é 90,71%; o reajuste Python verificado é 90,56%. Não substituir silenciosamente o primeiro pelo segundo. Falta identificar a execução e as previsões correspondentes usadas em cada figura e teste clássico.

### McNemar: mesma família de seis comparações

| Comparação | p nominal corrigido | p após Holm |
|---|---:|---:|
| CNN vs. VGG16 | 1,963e-5 | 3,926e-5 |
| CNN vs. VGG16 + LightGBM | 2,990e-7 | 8,969e-7 |
| CNN vs. RadImageNet | 6,056e-29 | 2,422e-28 |
| VGG16 vs. VGG16 + LightGBM | 0,182422 | 0,182422 |
| VGG16 vs. RadImageNet | 1,800e-37 | 9,001e-37 |
| VGG16 + LightGBM vs. RadImageNet | 3,532e-40 | 2,119e-39 |

Os três contrastes RadImageNet no PDF estão desatualizados. A atualização não muda as conclusões de significância. A comparação VGG16/híbrido mantém-se não significativa; isso não é uma demonstração de equivalência entre modelos. O híbrido é uma seleção exploratória entre sete classificadores avaliados no teste, e Holm não elimina esse viés.

## 5. Cruzamento com os oito comentários do revisor 1

| Comentário | Estado e ação proposta |
|---|---|
| 1 — Curadoria e efeito do leakage | A resposta reconhece que os splits mudaram, mas ainda garante um limiar conservador para imagens indistinguíveis. Retirar a garantia, mostrar distâncias e exemplos, e decidir a validação adicional. Apenas mudar a redação não valida as exclusões. |
| 2 — EfficientNet / BatchNorm | O texto já reconhece mudanças simultâneas de treino. Manter a melhoria atribuída ao protocolo completo, sem efeito causal específico de BatchNorm. |
| 3 — ImageNet / RadImageNet | Manter a comparação restrita às configurações avaliadas. Confirmar checkpoint e pré-processamento antes de uma conclusão sobre pretraining. Corrigir a ordem invertida de 72,1% / 97,5% na discussão: ImageNet é 97,49%; RadImageNet é 72,12%. |
| 4 — Escolha do híbrido no teste | A seleção best-of-seven já está reconhecida. Manter resultados e testes como exploratórios, sem apresentar 98,67% como estimativa confirmatória sem viés de seleção. |
| 5 — Independência entre pacientes | A falta de identificadores já está reconhecida. Descrever o filtro e a partição; não garantir independência por paciente nem controlo completo de duplicados enquanto a curadoria não estiver validada. |
| 6 — Estatística e IC | Corrigir o IC CNN e os contrastes RadImageNet. A resposta promete IC para todos os modelos principais, mas o clássico e ConvNeXt continuam sem IC na A1. Completar com a execução correta ou ajustar a resposta. |
| 7 — Consolidação das tabelas | A A1 foi consolidada no suplemento; continua repetição nas tabelas principais. Corrigir os números e decidir se a tabela consolidada deve passar para o corpo do artigo. |
| 8 — Resultados na Discussão | EfficientNet e a CNN adicional já aparecem nos Resultados. Rever os resultados quantitativos de explicabilidade ainda introduzidos na Discussão e atualizar a referência da resposta: a tabela EfficientNet é agora a Table 4. |

## 6. Comentários em inglês para colocar no documento

### Curadoria pHash — Methods, “Dataset and forensic audit”

I suggest revisiting this statement: we found visually distinct image pairs among the pHash exclusions, including a pair with a Hamming distance of 2. We therefore cannot guarantee that the threshold identifies only visually indistinguishable images. We should distinguish exact duplicates from images flagged by similarity and agree on the pair review and sensitivity analysis to present to Reviewer 1. The results were reproduced on the partition used in the study, but we do not yet know how revised curation would affect them.

### IC CNN — Tables 1 and A1

I propose updating the CNN confidence interval in Tables 1 and A1 to [91.89%, 95.43%], calculated from the verified predictions corresponding to the reported 93.81% accuracy using 1000 class-stratified bootstrap resamples (seed 42). The accuracy, F1 and AUC would remain unchanged. We should also document the bootstrap procedure so that the interval can be reproduced.

### RadImageNet — Table B11

The three McNemar comparisons involving RadImageNet do not match the verified final predictions. We should regenerate these comparisons and their Holm-adjusted p-values from the same identified prediction tables. The VGG16 softmax versus hybrid comparison remains non-significant, with p approximately 0.182.

### Modelo clássico — Table 1 e Limitations

We should distinguish the original classical-model result of 90.71% from the Python refit, which produced 90.56% in the verified run. Please confirm which execution supplied the predictions used for the confusion matrix and paired tests, so that metrics and statistical comparisons are not presented as belonging to the same execution unless they do.

### Valores invertidos — “Medical vs. natural pretraining”

The accuracy values appear in the opposite order to the models named in this sentence. ImageNet ResNet50 achieved 97.49%, while RadImageNet ResNet50 achieved 72.12%. We should reverse the values here.

### IC ausentes — Table A1 e resposta ao comentário 6

The response to Reviewer 1 states that confidence intervals are provided for all principal models, but the classical model and ConvNeXt still have missing intervals in Table A1. We should either provide the corresponding intervals or revise the response to describe exactly what has been added.

## 7. Redação possível se não forem feitas novas experiências

Uma limitação explícita é cientificamente mais defensável do que manter a garantia de exclusões válidas. Não sabemos se será suficiente para o revisor; esta alternativa não substitui a validação empírica pedida.

### Proposta para Limitations

The filtering procedure removed 50 exact-file duplicates and 909 images flagged by a pHash similarity rule. The pHash-based exclusions were not fully validated through manual review, and subsequent inspection identified visually distinct images among the pairs used for exclusion. The procedure may therefore have removed non-duplicate images, potentially affecting dataset composition and performance estimates. The reported results apply to the filtered partition used in this study; the effect of revised curation has not been assessed. Image-similarity filtering also does not establish patient- or study-level independence.

### Proposta para substituir o final da resposta ao comentário 1

We agree that the pHash threshold requires empirical validation rather than justification based on its numerical value alone. Subsequent inspection identified visually distinct images among the flagged pairs, including a pair with Hamming distance 2. The threshold should therefore be described as an image-similarity screening rule, rather than a validated criterion for identifying visually indistinguishable duplicates. The reported results correspond to the filtered partition used in the study; the effect of revising its exclusions has not yet been established.

Estes parágrafos não afirmam que foi feita uma revisão humana completa ou uma nova comparação de accuracies. Antes da submissão, é necessário alinhar Methods, título, resumo, discussão, conclusão e resposta ao comentário 5. Retirar “959 leaked images”, “visually indistinguishable” e formulações que garantam um benchmark livre de todas as dependências. Usar uma descrição da partição e do filtro efetivamente aplicados.

## 8. Decisões a tomar com os professores

1. Aplicar as correções estatísticas já sustentadas pelas previsões verificadas e resolver a identificação da execução clássica.
2. Decidir se a revisão incluirá apenas uma limitação explícita ou também adjudicação dos pares. Uma mudança de limiar sem validação não resolve os exemplos encontrados.
3. Se houver adjudicação, rever primeiro os pares interclasse e intersplit, verificando todas as ligações de cada imagem. Uma ligação falsa não prova que a imagem não tenha outra cópia verdadeira.
4. Versionar qualquer partição revista. Se mudar o treino, repetir os treinos e a extração de features dos métodos comparados; se mudar apenas o teste, a inferência dos checkpoints existentes permite uma primeira comparação, desde que a composição e origem das novas imagens sejam verificadas.
5. Se o objetivo for medir causalmente o efeito do leakage, definir um teste fixo e um contraste de treino com/sem cópias confirmadas, controlando também tamanho e composição do treino. A comparação histórica não isola esse efeito.

## 9. Evidência e reprodução

Fontes textuais: Manuscript.pdf, 29 páginas, e main.pdf, 11 páginas, correspondentes ao manuscrito revisto e à resposta ao revisor 1. As passagens foram também verificadas em leitura dos projetos Prism. Referências estáveis: Methods, Tables 1/A1/B11, Discussion e respostas aos comentários 1–8. Os PDFs de trabalho e links de acesso Prism não são distribuídos neste relatório público.

Ficheiros do repositório que sustentam os números:

- `data/manifests/audit_log_959.csv`: exclusões históricas recuperadas.
- `data/manifests/dataset_manifest_verified.csv`: hashes e inclusão das 6000 imagens.
- `results/curation/phash_candidates_6000.csv`: candidatos, igualdade RGB, prioridades e decisões por preencher.
- `results/curation/phash_summary.json`: contagens e sensibilidade do número de pares ao limiar.
- `results/curation/pixel_comparison_summary.json`: grupos de imagens RGB idênticas.
- `results/curation/original_manifest_check.json`: verificação do corpus original.
- `results/curation/execution_20261006.json`: estado concluído, versões e hashes dos scripts/entradas da execução.
- `results/corrected/metrics.csv` e `results/corrected/mcnemar.csv`: estatística dos modelos verificados.
- `docs/AUDIT_2026-10-05.md` e `docs/PHASH_REVIEW.md`: diagnóstico e detalhes metodológicos.

Comando portátil para repetir a auditoria, num ambiente com as dependências de análise instaladas:

```
python scripts/run_phash_audit.py --dataset /path/to/original/classification_task --output runs/new_phash_audit
```

O diretório de saída deve ser novo. A rotina verifica os ficheiros, gera candidatos e compara pixels; não decide exclusões e não altera o dataset.
