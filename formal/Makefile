COQC := $(shell command -v coqc 2>/dev/null)
ROCQ := $(shell command -v rocq 2>/dev/null)

# In dependency order.
FILES := GateIsa.v GateForm.v GateDomain.v GateMeaning.v GateSoundRange.v GateSoundNorm.v \
	GateSoundOp.v GateSoundRefine.v GateSoundAccess.v GateSoundJoin.v GateSoundLoad.v GateSoundLocal.v \
	GateSoundTransfer.v GateThread.v GateInterval.v GateCertificate.v GateBarrier.v GateLoop.v GateTerm.v GateContext.v GateEnds.v GateShared.v GateFront.v GateZero.v GatePrologue.v GateParse.v CatalogStep.v StoreLaunch.v StoreLink.v \
	StoreModel.v PipelineIsolation.v GateParticipation.v PipelineGrants.v

.PHONY: build check clean

# Compile an immutable snapshot, then publish it atomically. Existing
# verifications keep the resolved snapshot they started with. Keep old
# snapshots until an explicit clean; they may still have running readers.
build:
	@set -e; \
	if [ -n "$(COQC)" ]; then compile="$(COQC)"; \
	elif [ -n "$(ROCQ)" ]; then compile="$(ROCQ) c"; \
	else echo "error: neither coqc nor rocq is installed" >&2; exit 1; fi; \
	mkdir -p build; \
	exec 9>build/.lock; flock 9; \
	stage=$$(mktemp -d "$(CURDIR)/build/library.XXXXXXXX"); \
	link="$(CURDIR)/build/current.$$$$"; \
	trap 'rm -f "$$link"; rm -rf "$$stage"' EXIT HUP INT TERM; \
	cp $(FILES) Makefile "$$stage/"; \
	(cd "$$stage"; for f in $(FILES); do $$compile -q -Q . OpGate $$f; done); \
	ln -s "$$(basename "$$stage")" "$$link"; \
	mv -Tf "$$link" build/current; \
	trap - EXIT HUP INT TERM

check: build

clean:
	rm -rf build
	rm -f *.vo *.vos *.vok *.glob .*.aux .lia.cache .nia.cache
