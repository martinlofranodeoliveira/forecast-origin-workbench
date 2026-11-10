# Forecast Origin Workbench

Versão de desenvolvimento **0.1.0**. SDK Python e CLI local para inspecionar
backtests com origens crescentes em uma série numérica completa e regularmente
espaçada. O scikit-learn ajusta um regressor fixo de boosting por histogramas;
persistência e uma referência sazonal explícita permanecem na comparação.

Cada horizonte treina apenas com alvos já observados na origem da previsão.
Bandas empíricas usam erros de previsões anteriores cujos alvos também já foram
observados. Não há promessa de cobertura nem escolha automática do modelo.

[English](README.md) · [Contrato](docs/contract.md)

## Instalar e executar

Tuple qualificado: Linux x86_64, CPython3.13.18, scikit-learn1.9.1, NumPy2.5.4 e
SciPy1.18.1. Outras plataformas/versões não foram qualificadas. Os locks exatos de
runtime e desenvolvimento são separados e conferidos por hash.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps .
forecast-origin-workbench demo --assert
forecast-origin-workbench demo --assert --format csv
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
forecast-origin-workbench backtest --input examples/constant.json
```

O demo original usa128 valores sintéticos e quatro origens fixas. `--assert`
confere valores literais de referência/índices/banda, inclusive sob Python `-O`.
Durante o uso não há rede, download, serviço, arquivo de modelo nem opção para
carregar estimadores ou código arbitrário.

A entrada fechada tem `schema: forecast-origin-workbench-input/1`, `values` e
`plan`, com `lag_count`, `horizon_count`, `origins`, `seasonal_period`. Campos
opcionais: `start_tick`, `step`, `unit`, `budget`. O exemplo contém os128 valores.
`--input -` lê stdin limitado; arquivos precisam ser regulares. Symlink na folha,
FIFO, diretório e URI remota são rejeitados. Chaves duplicadas/desconhecidas,
números JSON não finitos e tipos incompatíveis têm erros estáveis.

## API Python

```python
from forecast_origin_workbench import BacktestPlan, Status, run_backtest, serialize_csv

result = run_backtest([7.0] * 128, BacktestPlan(1, 1, (64,), 3))
if result.status != Status.COMPLETE:
    raise RuntimeError(result.reason)
encoded = serialize_csv(result)
assert encoded.status == Status.COMPLETE
print(len(encoded.data))  # 734 bytes, incluindo LF neste caso literal
```

Plano/orçamento são metadados imutáveis passivos, validados por `run_backtest`.
Os registros retornados são imutáveis; modelos ficam privados em memória.

## Interpretar o resultado

Na origem `o`, os dados até `y[o]` já existem. O treino do horizonte `h` admite
âncora `a` apenas se `a+h <= o`. Features são os últimos `L` valores observados.
Cada origem/horizonte ajusta um HGB direto de64 iterações, sem early stopping,
busca de parâmetros, previsão recursiva ou repetição de sementes. Apenas o HGB
possui campos de treino; persistência/sazonal usam `NOT_APPLICABLE`, campos de
treino null e o índice observado que realmente sustenta a referência.

Uma previsão anterior da origem `r` só fornece erro quando `r < o` e `r+h <= o`.
Com menos de3 erros, a banda é `UNAVAILABLE_INSUFFICIENT_HISTORY`, com raio e
extremos null. Nos demais casos usa o rank inteiro `ceil(4m/5)` dos erros
absolutos. Erros suficientes iguais a zero dão banda **disponível** de raio0.
Há no máximo7 resíduos neste escopo. O parâmetro4/5 não afirma cobertura80%,
confiança conformal ou independência. Mudanças no processo podem invalidar a
utilidade dos erros passados.

MAE/RMSE mantêm as unidades sintéticas. A média geral dá o mesmo peso a cada
célula, embora horizontes possam compartilhar o mesmo alvo. O modelo aprendido
pode perder das referências. JSON preserva plano, hashes, linhagem, previsões,
alvos, erros e métricas; CSV contém as linhas de previsão/score. Guarde JSON para
manter o contexto completo de unidades e proveniência.

Limites:128–4096 valores finitos, magnitude até1000000, sem ausência/bools;
até32 lags,8 horizontes,8 origens e64 pares mínimos por ajuste. São64 ajustes
HGB e192 células no máximo. Período sazonal explícito2–32. Entrada JSON até512KiB.
JSON canônico do SDK e cada JSON/CSV emitido têm limites próprios de1MiB,
contando LF; formatos não são somados. `--max-output-bytes` reduz o limite do
formato escolhido. Falhas de entrada/SDK prevalecem mesmo com limite CLI zero.
O orçamento independente `budget.max_json_bytes=0` do SDK continua interrompendo
antes de copiar valores ou ajustar modelos. JSON canônico precisa caber antes de `COMPLETE`, mesmo
quando a saída escolhida é CSV. Contadores não garantem RSS ou prazo universal.

Comandos normais escrevem JSON/CSV completo em stdout; erros usam JSON fechado
em stderr, sem caminho/valor de entrada ou mensagem arbitrária de dependência.
Saídas:0 completo/ajuda,2 entrada/plano/uso inválido,3 orçamento,4 falha de
ajuste/execução/I/O,5 indeterminação numérica. `FIT_FAILED` e `INCOMPLETE` são
estados distintos. OOM/kill pode impedir qualquer resultado. Mesmo com bytes
preparados antes, stdout pode falhar após um prefixo; descarte a saída se o
código não for0. Se stderr também falhar, o diagnóstico pode não chegar; o código não zero continua indicando falha. Redirecionamento do shell pode criar/truncar destino antes do
programa.

## Verificação reproduzível

```sh
sh scripts/check.sh
```

Só a preparação explícita baixa wheels oficiais conferidos por hash. Testes e
runtime novo usam network-none, nenhum bind do host, UID/GID1000, capacidades
removidas, NNP, PID256,1GiB e2CPU. O consumidor do wheel reconstruído não tem
checkout/compilador/backend e confere ajuste real, índices temporais, maturidade,
Decimal, mutações do futuro, bytes e CLI instalada. Evidências ficam em `dist/`
ignorado. Cada gate primeiro remove o `dist/` gerado deste projeto para não
reter evidência de uma execução anterior quando há falha. Symlink ou arquivo
não diretório nesse caminho é rejeitado sem remoção. Gates locais e CI hospedada da fonte exata são evidências distintas;
um YAML não comprova execução hospedada.

MIT cobre somente o código original Lofrano. Dependências/bibliotecas nativas
mantêm termos próprios em [avisos](THIRD_PARTY_NOTICES.md) e `_notices` no pacote.
[Lofrano Tecnologia](https://lofranotecnologia.com.br).
