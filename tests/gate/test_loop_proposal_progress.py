"""The untrusted proposer stops at edges its DAG validator rejects."""

import subprocess

import pytest

from opgate.gate import formal_dir

HEADER = "From Coq Require Import List ZArith Bool.\nFrom OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.\nImport ListNotations.\n"


def compile_query(tmp_path, query):
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


LEGACY_AND_PROOF = "Fixpoint legacy_loop_dag_collect (fuel:nat)(F:lfacts)(prog:program)(endpoint:pc)\n (pending:list(pc*option astate)) : option(list astate) :=\n match pending with\n | []=>Some []\n | (pos,ost)::tail=>match fuel with\n   | O=>None\n   | S rest=>match ost with\n     | None=>legacy_loop_dag_collect rest F prog endpoint tail\n     | Some st=>if is_bottom st then legacy_loop_dag_collect rest F prog endpoint tail else\n       if phase_event endpoint pos then\n         match legacy_loop_dag_collect rest F prog endpoint tail with Some rest=>Some(st::rest)|None=>None end\n       else match fetch prog(fst pos)(snd pos)with\n         | Some i=>legacy_loop_dag_collect rest F prog endpoint(loop_dag_successors F prog pos st i++tail)\n         | None=>None end\n     end\n   end\n end.\nTheorem collector_preserves_checked : forall fuel F prog protected events endpoint post pending,\n loop_dag_all fuel F prog protected events endpoint post pending = true ->\n loop_dag_collect fuel F prog endpoint pending = legacy_loop_dag_collect fuel F prog endpoint pending.\nProof.\n induction fuel as [|fuel IH]; intros F prog protected events endpoint post pending H.\n - destruct pending as [|[pos ost] tail]; [reflexivity|discriminate].\n - destruct pending as [|[pos ost] tail]; [reflexivity|].\n   destruct ost as [st|].\n   + cbn [loop_dag_all loop_dag_collect legacy_loop_dag_collect] in *.\n     destruct (is_bottom st); [eapply IH;exact H|].\n     destruct (phase_event endpoint pos).\n     * destruct (state_le F st post); [|discriminate].\n       rewrite (IH F prog protected events endpoint post tail H). reflexivity.\n     * destruct (occurrence_member events pos); [discriminate|].\n       destruct (fetch prog (fst pos) (snd pos)) as [i|]; [|discriminate].\n       destruct (wf_instr F i && short_all (fun r => loop_preserves r i) protected && loop_dag_forward prog pos i) eqn:Hi; [|discriminate].\n       apply andb_true_iff in Hi. destruct Hi as [_ Hforward].\n       rewrite Hforward. eapply IH;exact H.\n   + cbn [loop_dag_all loop_dag_collect legacy_loop_dag_collect] in *.\n     eapply IH;exact H.\nQed.\nPrint Assumptions collector_preserves_checked.\n"


def test_all_successful_dag_checks_preserve_proposals(tmp_path):
    result = compile_query(tmp_path, HEADER + LEGACY_AND_PROOF)
    assert result.stdout.count("Closed under the global context") == 1
    assert "Axioms:" not in result.stdout


METADATA = r"""
Definition F : lfacts := {| lf_widths:=PM.empty nat;lf_fixed:=[];lf_bounds:=[];
 lf_aligns:=[];lf_roles:=[];lf_shared_window:=1%positive;lf_local_window:=2%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[] |}.
Definition st:=empty_state F.
Definition p:program:=[[IBra None 0%nat];[INop]].
"""


@pytest.mark.parametrize(
    "proposition",
    [
        "loop_dag_collect 10000 F p (1%nat,0%nat)[((0%nat,0%nat),Some st)]=None",
        "loop_dag_collect 1 F p (0%nat,0%nat)[((0%nat,0%nat),Some st)]=Some( st::nil)",
        "loop_dag_collect 0 F p (1%nat,0%nat)[]=Some []",
        "loop_dag_collect 1 F p (1%nat,0%nat)[((0%nat,0%nat),None)]=Some []",
        "loop_dag_collect 2 F [[IBra None 1%nat];[INop]](1%nat,0%nat)[((0%nat,0%nat),Some st)]=Some[st]",
        "loop_dag_collect 2 F [[IExit None];[INop]](1%nat,0%nat)[((0%nat,0%nat),Some st)]=None",
        "loop_dag_collect 2 F p(1%nat,0%nat)[((4%nat,0%nat),Some st)]=None",
        "loop_dag_collect 2 F [[IBra(Some(false,3%positive))0%nat];[INop]](1%nat,0%nat)[((0%nat,0%nat),Some st)]=None",
        "loop_dag_collect 1 F p(1%nat,0%nat)[((0%nat,0%nat),Some bottom_state)]=Some []",
    ],
)
def test_proposal_progress_boundaries(tmp_path, proposition):
    compile_query(
        tmp_path,
        HEADER
        + METADATA
        + "Example checked: "
        + proposition
        + ".\nProof. vm_compute. reflexivity. Qed.\n",
    )
