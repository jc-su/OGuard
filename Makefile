# Reproduce OpGate's evaluation: each experiment writes a record to eval/records/,
# `make export` turns the records into paper/figures/data/*.csv, the generated
# tables, and paper/figures/numbers.tex, and `make paper` draws the figures from
# them. `make help` lists the targets; eval/README.md says what each one answers.
#
# Timed experiments run on an otherwise idle GPU, pinned to one core of that
# GPU's NUMA node (eval/timing.py), and write RUNS runs in fresh processes;
# the multi-process consolidation experiment runs on the cores NODE of that node.
# The registration experiment's proofs run on PROOF_CORES, off the GPU's node,
# so that it can run beside the timed experiments.
# Set GPU and CORE for the machine, and UPSTREAM to a DataKernelBench checkout at
# commit 43c361d4ed789d1feaffbcee750e5b256590571a for the experiments that read it.

GPU ?= 1
CORE ?= 6
NODE ?= 0-31
PROOF_CORES ?= 32-63
GUARD_COST_RUNS := 2
COST_CORES ?= 32,33
COST_KERNELS ?= sonnet/q6_sum/numba sonnet/gather/numba haiku/bloom_probe/numba haiku/merge_sorted/numba
DATE ?= $(shell date +%F)
UPSTREAM ?= /tmp/opgate-datakernelbench
REPAIRS ?= $(RECORDS)/repairs_$(DATE).json
ABLATION ?= $(RECORDS)/rule_ablation_$(DATE).json
REGISTRATION ?= $(RECORDS)/registration_$(DATE).json
GUARDED ?= $(RECORDS)/guarded_2026-10-07b.json
ARTIFACT ?= /tmp/opgate-artifact.tar.gz
RECORDS := eval/records

RUN := uv run python -m
ON_GPU := CUDA_VISIBLE_DEVICES=$(GPU) uv run python -m
TIMED := CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(CORE) uv run python -m
RUNS := $(TIMED) eval.timing

.PHONY: help check test test-cpu-full test-all export paper artifact \
	motivation motivation-cudf-udf motivation-dask-worker motivation-index-intermediates \
	motivation-generated-kernel \
	rq1 rq1-corpus rq1-corpus-from-source rq1-corpus-ptx rq1-generated-ptx rq1-paired-replay rq1-service-boundary rq1-trace-replay rq1-guarded-defects rq1-leakage \
	rq2-rule-ablation rq2-acceptance rq2-plan-support rq2-cudf-udfs rq2-repair rq2-guarded rq2-differential \
	rq3 rq3-service-pipeline rq3-service-q3 rq3-index-path rq3-tpch-queries rq3-torch-queries \
	rq4 rq4-registration rq4-registration-cost rq4-step-check rq4-clearing rq4-kernel-time rq4-query-overhead rq4-guard-cost \
	rq4-consolidation rq4-mixed rq4-density rq4-fault-isolation rq4-constraint-generation rq4-generation \
	rq4-cold-query timed

help:
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 34

# ---------------------------------------------------------------- the code

check: ## format, lint, and type-check without changing files
	uv run ruff format --check
	uv run ruff check
	uv run ty check

test: ## fast CPU regressions; excludes explicitly marked slow and GPU tests
	uv run pytest -m 'not slow and not gpu' -q

test-cpu-full: ## all CPU tests, including whole-kernel proof and boundary regressions
	uv run pytest -m 'not gpu' -q

test-all: ## every test, including the slow GPU tests
	uv run pytest -q

# ---------------------------------------------------------------- motivation

motivation: motivation-cudf-udf motivation-dask-worker motivation-index-intermediates motivation-generated-kernel ## every motivation experiment

motivation-cudf-udf: ## a cuDF UDF reads another tenant's column, and a wild load ends the context
	$(ON_GPU) eval.motivation.cudf_udf --output $(RECORDS)/cudf_udf_$(DATE).json

motivation-dask-worker: ## two Dask clients on one dask-cuda worker: a read of the other's column, and a fault
	$(ON_GPU) eval.motivation.dask_worker --output $(RECORDS)/dask_worker_$(DATE).json

motivation-index-intermediates: ## index-valued intermediates in the 22 published TPC-H plans
	$(ON_GPU) eval.motivation.index_intermediates --upstream $(UPSTREAM) --output $(RECORDS)/index_intermediates_$(DATE).json

motivation-generated-kernel: ## a generated TPC-H Q3 module faults a shared Dask worker
	$(ON_GPU) eval.motivation.generated_kernel --upstream $(UPSTREAM) --output $(RECORDS)/generated_kernel_$(DATE).json

# ---------------------------------------------------------------- RQ1: safety

rq1: rq1-corpus rq1-corpus-from-source rq1-corpus-ptx rq1-generated-ptx rq1-paired-replay rq1-service-boundary rq1-trace-replay rq1-guarded-defects ## every RQ1 experiment

rq1-corpus: ## the gate's decision on every corpus kernel's recorded PTX against the reviewed tables
	$(RUN) eval.corpus.run

rq1-corpus-from-source: ## every corpus kernel compiled from its source and registered with the executor
	$(ON_GPU) eval.corpus.run --from-source

rq1-corpus-ptx: ## the gate on the PTX of the 27 defective sources that an unchecked run triggered
	$(RUN) eval.rq1_safety.corpus_ptx --output $(RECORDS)/corpus_ptx_$(DATE).json

rq1-generated-ptx: ## the gate on the compiled PTX of the generated TPC-H Q3 module
	$(RUN) eval.rq1_safety.generated_ptx --upstream $(UPSTREAM) --output $(RECORDS)/generated_ptx_$(DATE).json

rq1-paired-replay: ## the same PTX launched unchecked and through the gate
	$(ON_GPU) eval.rq1_safety.paired_replay --output $(RECORDS)/paired_replay_$(DATE).json

rq1-service-boundary: ## unsafe PTX, handles, and queries through the multi-tenant service
	$(ON_GPU) eval.rq1_safety.service_boundary --output $(RECORDS)/service_boundary_$(DATE).json

rq1-trace-replay: ## Rocq's check_step on every step that the catalog decided in five workloads
	$(ON_GPU) eval.rq1_safety.trace_replay --output $(RECORDS)/trace_replay_$(DATE).json

rq1-guarded-defects: ## the triggered corpus defects with their unproved accesses checked, beside another tenant
	$(ON_GPU) eval.rq1_safety.guarded_defects --output $(RECORDS)/guarded_defects_$(DATE).json

rq1-leakage: ## does each system expose one tenant's data to another tenant's kernel on one GPU (shared, MPS, Guardian, OpGate)
	$(ON_GPU) eval.rq1_safety.leakage --output $(RECORDS)/leakage_$(DATE).json

# ---------------------------------------------------------------- RQ2: coverage

rq2-rule-ablation: ## the accepted kernels that each group of the checker's rules is needed for (reads REGISTRATION)
	CUDA_VISIBLE_DEVICES= taskset -c $(PROOF_CORES) $(RUN) eval.rq2_coverage.rule_ablation \
		--registration $(REGISTRATION) --output $(ABLATION) --workers 32 --timeout 600

rq2-acceptance: ## the gate's acceptance on the corpora and in the repair loop (reads REPAIRS)
	$(RUN) eval.rq2_coverage.acceptance --repairs $(REPAIRS) --output $(RECORDS)/acceptance_$(DATE).json

rq2-plan-support: ## static acceptance by represented query plan (reads current REGISTRATION)
	$(RUN) eval.rq2_coverage.plan_support --registration $(REGISTRATION) \
		--output $(RECORDS)/query_plan_support_$(DATE).json

rq2-cudf-udfs: ## the gate on the examples of cuDF's guide to UDFs
	$(ON_GPU) eval.rq2_coverage.cudf_udfs --output $(RECORDS)/cudf_udfs_$(DATE).json

rq2-repair: ## the refusal-guided repair loop over the refused kernels
	$(ON_GPU) eval.corpus.repair --output $(RECORDS)/repairs_$(DATE).json

rq2-guarded: ## every kernel with its unproved accesses checked when they run (reads REGISTRATION)
	taskset -c $(PROOF_CORES) $(RUN) eval.rq2_coverage.guarded --registration $(REGISTRATION) --every \
		--output $(RECORDS)/guarded_$(DATE).json

rq2-differential: ## safe guard-changed kernels, as written and as the gate accepts them, on their workloads
	$(ON_GPU) eval.rq2_coverage.differential --guarded $(GUARDED) --output $(RECORDS)/differential_$(DATE).json

# ---------------------------------------------------------------- RQ3: pipelines

rq3: rq3-index-path rq3-tpch-queries rq3-torch-queries ## the timed RQ3 experiments

rq3-service-pipeline: ## verified index producer and gather composed with a trusted cuDF group-by
	$(ON_GPU) eval.rq3_pipelines.service_pipeline --output $(RECORDS)/service_pipeline_$(DATE).json

rq3-service-q3: ## complete mixed Q3 service query with a verified generated revenue projection
	$(ON_GPU) eval.rq3_pipelines.service_q3 --output $(RECORDS)/service_q3_$(DATE).json

rq3-index-path: ## the static index path against the guarded one, by selectivity and scale (timed)
	$(RUNS) eval.rq3_pipelines.index_path $(RECORDS)/index_path_$(DATE)

rq3-tpch-queries: ## the 22 TPC-H queries on cuDF, with every allocation through the catalog (timed)
	$(RUNS) eval.rq3_pipelines.tpch_queries $(RECORDS)/tpch_queries_sf1_$(DATE) --scale 1
	$(RUNS) eval.rq3_pipelines.tpch_queries $(RECORDS)/tpch_queries_sf10_$(DATE) --scale 10

rq3-torch-queries: ## TPC-H Q1, Q6, and Q14 on a PyTorch tensor engine with model-written operators (timed)
	$(RUNS) eval.rq3_pipelines.torch_queries $(RECORDS)/torch_queries_sf1_$(DATE) --scale 1
	$(RUNS) eval.rq3_pipelines.torch_queries $(RECORDS)/torch_queries_sf10_$(DATE) --scale 10

# ---------------------------------------------------------------- RQ4: cost

rq4: rq4-step-check rq4-clearing rq4-kernel-time rq4-query-overhead ## the timed RQ4 microbenchmarks

rq4-registration: ## the final gate's decision and time for every corpus kernel, 8 at a time
	CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(PROOF_CORES) uv run python -m eval.rq4_cost.registration \
		--output $(REGISTRATION)

rq4-registration-cost: ## fresh-process cold checks and exact same-process reuse on COST_KERNELS
	taskset -c $(COST_CORES) uv run python -m eval.rq4_cost.registration_cost \
		$(foreach kernel,$(COST_KERNELS),--kernel $(kernel)) \
		--output $(RECORDS)/registration_cost_$(DATE).json

rq4-step-check: ## the step check against the live buffers (timed)
	$(RUNS) eval.rq4_cost.step_check $(RECORDS)/step_check_$(DATE)

rq4-clearing: ## clearing a region that another tenant owned (timed)
	$(RUNS) eval.rq4_cost.clearing $(RECORDS)/clearing_$(DATE)

rq4-kernel-time: ## an accepted kernel at NVVM level 1 against level 3 (timed)
	$(RUNS) eval.rq4_cost.kernel_time $(RECORDS)/kernel_time_$(DATE)

rq4-guard-cost: ## guarded kernels against the kernels as written: outputs and device time (timed)
	$(RUNS) --runs $(GUARD_COST_RUNS) eval.rq4_cost.guard_cost $(RECORDS)/guard_cost_$(DATE) --registration $(REGISTRATION) \
		--proof-cores $(PROOF_CORES)

rq4-query-overhead: ## a query through OpGate against direct launches and cuDF, at six scales (timed)
	$(RUNS) eval.rq4_cost.query_overhead $(RECORDS)/query_overhead_$(DATE)

rq4-consolidation: ## 1 to 16 tenants through the service, a service or a process each, MPS with private or shared tables, and Guardian (timed)
	CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(NODE) uv run python -m eval.rq4_cost.consolidation \
		--output $(RECORDS)/consolidation_$(DATE).json

rq4-mixed: ## 1 to 16 tenants running eight templates of model-written operators through the service, with refused operators skipped or served by built-in operators, and under MPS (reads REGISTRATION and GUARDED; timed)
	CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(NODE) uv run python -m eval.rq4_cost.mixed \
		--registration $(REGISTRATION) --guarded $(GUARDED) --proof-cores $(PROOF_CORES) \
		--output $(RECORDS)/mixed_$(DATE).json

rq4-density: ## tenants of one scale-factor-10 table that each configuration holds (timed)
	CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(NODE) uv run python -m eval.rq4_cost.density \
		--output $(RECORDS)/density_$(DATE).json

rq4-fault-isolation: ## one tenant's faulting kernel against the others: shared context, process, MPS, and the service (reads GPU)
	CUDA_VISIBLE_DEVICES=$(GPU) taskset -c $(NODE) uv run python -m eval.rq4_cost.fault_isolation \
		--output $(RECORDS)/fault_isolation_$(DATE).json

rq4-generation: ## the front end's compile of every corpus kernel to PTX, each in a fresh process, 16 at a time
	CUDA_VISIBLE_DEVICES= taskset -c $(PROOF_CORES) uv run python -m eval.rq4_cost.generation \
		--output $(RECORDS)/generation_$(DATE).json

rq4-cold-query: ## TPC-H Q1, Q6, and Q14 on the PyTorch engine from a fresh process: compile, decision, load, first run, and warm (timed)
	$(RUNS) eval.rq4_cost.cold_query $(RECORDS)/cold_query_$(DATE) --scale 1

rq4-constraint-generation: ## the checker's bounded constraint generation against the eager one
	$(RUN) eval.rq4_cost.constraint_generation --output $(RECORDS)/constraint_generation_$(DATE).json

timed: rq3 rq4 rq4-consolidation ## every timed experiment, one after another

# ---------------------------------------------------------------- the paper

export: ## the records to paper/figures/data/*.csv, the tables, and numbers.tex
	$(RUN) eval.export

paper: export ## the paper, with its figures drawn from the exported data
	$(MAKE) -C paper

artifact: ## the anonymous review archive at ARTIFACT, which fails on identifying text
	$(RUN) eval.artifact --output $(ARTIFACT)
