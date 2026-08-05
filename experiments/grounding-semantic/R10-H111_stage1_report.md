# R10-H111 stage 1 - generation report

Model facebook/mbart-large-50, p=0.2, referee v3 thresholds: {"nll_max": 6.234306645393371, "distinct3_min": 0.9129795396419437, "maxrun_max": 1.0, "charrun_max": 3.0, "symdens_max": 0.06910171017101704}
Reconstructions consumed: 260452; admitted drift 83714; paraphrase augmentation 12606 {'procedural': 6583, 'quantitative': 2860, 'scientific': 3163}

## Counts

shape: (6, 3)
┌───────────────────────┬───────┬───────┐
│ tag                   ┆ label ┆ len   │
│ ---                   ┆ ---   ┆ ---   │
│ str                   ┆ i64   ┆ u32   │
╞═══════════════════════╪═══════╪═══════╡
│ ae_drift_procedural   ┆ 0     ┆ 39832 │
│ ae_drift_quantitative ┆ 0     ┆ 17549 │
│ ae_drift_scientific   ┆ 0     ┆ 26333 │
│ ae_para_procedural    ┆ 1     ┆ 6583  │
│ ae_para_quantitative  ┆ 1     ┆ 2860  │
│ ae_para_scientific    ┆ 1     ┆ 3163  │
└───────────────────────┴───────┴───────┘

## Eyeball - 50 admitted drift (main-session precision adjudication)

**D1** [ae_drift_procedural] min-entail 0.01
- seed: If `openpyxl <https://openpyxl.readthedocs.io/en/stable/>`__ is installed, many of these cases will now default to using the ``openpyxl`` engine.
- rec : If `openpyxl <https://openpyxlxl.openpyxlxls.readtheme-only.ss. <https://://open.openpyxl.

**D2** [ae_drift_scientific] min-entail 0.01
- seed: Since then, the technique has evolved to include experiments utilising lower frequencies, wider bandwidths, and entire arrays of antenna.
- rec : Since then, the technique has evolved to include the following: the technique has to the extent, and wider bandwidths, and entire arrays of antenna.

**D3** [ae_drift_procedural] min-entail 0.39
- seed: Have you sold the home you bought with the prior loan and have paid that loan in full?
- rec : Have you sold the property you bought the home with the home you bought and the loan in full?

**D4** [ae_drift_scientific] min-entail 0.00
- seed: Kernel-based nonparametric hazard rate estimation is considered with a special class of infinite-order kernels that achieves favorable bias and mean square error properties.
- rec : Kernel--Rato the non-based non-Ae-based nonpara..., is considered with a result of a single--based non--a.

**D5** [ae_drift_procedural] min-entail 0.00
- seed: yes, It s quick and easy to sign up for the card . Call the toll - free Direct Express hotline at 1 - 800 - 333 - 1795 . Also, Social Security can help you sign up .
- rec : yes, It s quick and easy to sign up for the card . Call the toll - free Direct Express .

**D6** [ae_drift_quantitative] min-entail 0.01
- seed: Bruce Fairbairn produced an album released in the nineties.
- rec : The first and second.

**D7** [ae_drift_scientific] min-entail 0.02
- seed: We present a new class of exact nonsingular solutions for the Maxwell equations in vacuum, which describe the electromagnetic field of the counterpropagating focused laser beams and the subperiod focused laser pulse.
- rec : We present a new class of exact nonsingular solutions for the Maxwell equations in the equations in the nonsingular solution of nonsing equations in vacuum.

**D8** [ae_drift_scientific] min-entail 0.00
- seed: We find two cases that are not transformable via the 'admissible transformations' to the lattice equations in the existing classification.
- rec : We are not transformable.

**D9** [ae_drift_scientific] min-entail 0.06
- seed: We evaluate a Gaussian entanglement measure for a symmetric two-mode Gaussian state of the quantum electromagnetic field in terms of its Bures distance to the set of all separable Gaussian states.
- rec : We evaluate a Gaussian entanglement of the distance to the state of the quantum entanglement.

**D10** [ae_drift_procedural] min-entail 0.00
- seed: Well, put in that way, certainly it could lead to that. For those drivers who have records of drinking and driving, it is mandatory for them to install an ignition interlock device, and that will be indicated on your documentation.
- rec : Well, and if you can also, you have to do in that way, you.

**D11** [ae_drift_procedural] min-entail 0.10
- seed: Based on the provided evidence, the design of Bavarian curling stones differs from Scottish curling stones in that the handle on a Bavarian stone is vertical, whereas the handle on a Scottish stone is horizontal.
- rec : Based on the provided evidence, the

**D12** [ae_drift_scientific] min-entail 0.01
- seed: For each of these two scenarios, we first derive the conditions to produce a GRB 980425-like event and we then discuss the consequences for the event rate.
- rec : For each of these two scenarios, we first derive the conditions to produce a GRB 980425--like event. We then discuss the consequences for the consequences.

**D13** [ae_drift_procedural] min-entail 0.02
- seed: ``` In file pandas/core/frame.py, add: ```python numpy.percentile ```
- rec : ``` In file pandas/-default/s/core/core/where/frame/default/core/core/core/core//core/core/frame/core/core/frame.py, add: ```

**D14** [ae_drift_procedural] min-entail 0.30
- seed: In file youtube_dl/PostProcessor.py, add: ```python class FFmpegMergerPP(FFmpegPostProcessor): def run(self, info): filename = info['filepath'] args = ['-c', 'copy'] self.run_ffmpeg_multiple_files(info['__files_to_merge'], filename, args) return True, info ``` In file youtube_dl/YoutubeDL.py, add: ```python prepend_extension, from .PostProcessor im
- rec : In file youtube_dl/Youtube_dl/PostProcessor/Youtube.python prepend_

**D15** [ae_drift_procedural] min-entail 0.01
- seed: The split-apply-combine combination rules attempt to be as common sense based as possible.
- rec : The split-apply-combine combinations are a simple, in-appr-a-combine-a-a-like-a-like-likely-to-be-and-like-a-a-in-and-like-a-like-a-a-like-a-like-like-a-re-like-a-like-a-like-a-a-a-a-like-like-a-like-a-like-alike

**D16** [ae_drift_procedural] min-entail 0.02
- seed: The failure block is located in `astropy/_dev/scm_version.py` at line 9, where an `ImportError` is raised with the message "setuptools_scm broken or not installed".
- rec : The failure block is located in `astropy/_dev/scm_d/scm_version.py` at line 9, where the message "setuptools.scm/sscm_version.py is not installed".

**D17** [ae_drift_scientific] min-entail 0.01
- seed: We investigate the dynamics of a continuous atom laser based on the merging of independently formed atomic condensates.
- rec : . the merging of the merging.

**D18** [ae_drift_quantitative] min-entail 0.01
- seed: The Canary Islands were colonized by Spain.
- rec : The Canary Islands were a.s.. and.

**D19** [ae_drift_procedural] min-entail 0.00
- seed: Yes, Guyana joined the Regional Security System in 2022.
- rec : Yes, Guyana - The Regional Security System is in Security.

**D20** [ae_drift_procedural] min-entail 0.32
- seed: Do you need more help regarding this topic?
- rec : Do you need more help with more help?

**D21** [ae_drift_scientific] min-entail 0.01
- seed: More recently, the Dirac and Einstein equations were unified in a tetrad formulation of a Kaluza-Klein model which gives precisely the usual Dirac-Einstein Lagrangian.
- rec : More recently, the Dirac and Einstein equations were unified in the formulation of a new equations, but it was, and it was the Dirac-Einstein Lagrangian.

**D22** [ae_drift_procedural] min-entail 0.01
- seed: Would you like to sign up for a MyDMV account?
- rec : Would you like to signify the?

**D23** [ae_drift_procedural] min-entail 0.06
- seed: Therefore, I cannot identify the code block that defines the little-h equivalency from this snippet.
- rec : Therefore, I cannot identify the code block that defines the little-h equivalency from this code block.

**D24** [ae_drift_scientific] min-entail 0.00
- seed: We also extend the result to a large family of quantifiers which includes the negativity, the robustness of entanglement, and the best separable approximation measure.
- rec : We also extend the result of the result to a largely to a large of the.

**D25** [ae_drift_procedural] min-entail 0.00
- seed: of course you may want to choose a funeral director to help you plan the funeral. Then you or the funeral director can call the National Cemetery Programming Office at 800 - 535 - 1117 to request a burial.
- rec : of course you may you may want to choose a funeral director to help you.

**D26** [ae_drift_scientific] min-entail 0.01
- seed: In this paper, we propose a geometric integrator for nonholonomic mechanical systems.
- rec : In this papersection we propose this paper, we propose a geometric integrator, we propose a geometric integrator for a non non-noncounting for non- the non-

**D27** [ae_drift_procedural] min-entail 0.01
- seed: To install jsdom, use the command `npm install jsdom`.
- rec : To install jsdom, use the following command to install jsdom`.

**D28** [ae_drift_scientific] min-entail 0.00
- seed: Aims: In this paper we study whether the shock-in-jet model, widely used to explain the outbursting behaviour of quasars, can be used to explain the radio flaring behaviour of the microquasar Cygnus X-3.
- rec : Aims: In this paper we study whether the shock-in-in-jet model, which the shock-in-jet model, widely used to explain the outburstings behaviour of quasars, can be used to explain the outbursting behaviour of the quasars.

**D29** [ae_drift_procedural] min-entail 0.00
- seed: It notes that using Qwen2-1.5B as the baseline results in an EMDM score range increase from 25.94% to 56.53%, and that Mistral 7B is ultimately selected as the baseline for the paper's experiments.
- rec : It notes that using Qwen2-1.5B as the baseline is, and that the baselineline is the baseline.

**D30** [ae_drift_scientific] min-entail 0.01
- seed: The p-detector is based on two tunnel junctions in a Aharonov-Bohm-type setup.
- rec : The p-detector is a game of a p-detectoral-type.

**D31** [ae_drift_procedural] min-entail 0.01
- seed: POS taggers have lower accuracy for adjectives primarily due to the confusion between adjectives (JJ/ADJ) and nouns (NN/NOUN).
- rec : POS (NN/NOun) NN/NOUN (NN/NO)

**D32** [ae_drift_scientific] min-entail 0.00
- seed: We study how the charges of the black rings measured at the asymptotic infinity are encoded in the near-horizon metric and gauge potentials, independent of the detailed structure of the connecting region.
- rec : We study how the black rings, study: the black . The metricss . The metric the . The as a. the detailed structure of thee at-sss of the , of the graphy structured.

**D33** [ae_drift_procedural] min-entail 0.00
- seed: Additionally, the data includes metadata indicating the original and translated languages, and while PoS-tagging is beneficial, it is not a strict requirement for the corpus construction described.
- rec : Additionally, the data is beneficial, and it is beneficiad, and beneficial.

**D34** [ae_drift_scientific] min-entail 0.01
- seed: Regular arrays are extremely resilient and can reversibly accommodate a large amount of supercoiling without much change in length.
- rec : Regular:--Rescale-like-a---a-and-s-on-a-a--a-a-s-a--s-s-a-s-s--a-a-s-s--on--a-s--a-s-s--a-s--s-s-

**D35** [ae_drift_procedural] min-entail 0.01
- seed: If you were on active duty in Vietnam, Thailand or the Korean Demilitarized Zone and you were exposed to specific chemicals during this time, and if as a result your child has spina bifida or other birth defects, then your child may be eligible for disability benefits
- rec : If you have a. If you have spind of the time you were exposed to the birth defects, then,

**D36** [ae_drift_quantitative] min-entail 0.00
- seed: the 25th episode had 4.4 million viewers
- rec : the 25th episode had a lot of a few moments, and had

**D37** [ae_drift_quantitative] min-entail 0.03
- seed: arnold palmer finished as runner - up to jack nicklaus in three tournaments
- rec : arnolded the runner-up the runner-up in three tournaments

**D38** [ae_drift_procedural] min-entail 0.00
- seed: According to a 2023 interview, the rapid progress in AI caused some of Hofstadter's "core beliefs" about AI's limitations to collapse.
- rec : According to a 2023-to-tradical---complete-up-they-total-comment-in-and-to-the-be--and-the-a-s--in--a-the-wrong--core-to--some-s-the-con-core-to-the-to-in-and-to-the-in-a-the-in--and-the-in-to-the--

**D39** [ae_drift_scientific] min-entail 0.05
- seed: The phase structure of neutral two-flavor quark matter at nonzero temperature is studied.
- rec : The phase structure of the phase--free-de-r.

**D40** [ae_drift_procedural] min-entail 0.00
- seed: In solar calendars, this is typically achieved by adding an extra day (leap day) to February in leap years, such as in the Julian and Gregorian calendars, or by including epagomenal days.
- rec : In solar calendars, this is typically achieved by adding an extra day (leap day (leap)) to the day of the day (leap) to the calendar.

**D41** [ae_drift_procedural] min-entail 0.37
- seed: There a Consumer Services Representative from the DMV will contact you in an attempt to resolve this complaint .
- rec : There a Consumer Services contact from the DMV will contact in a DMV will in an attempt to resolve this complaint .

**D42** [ae_drift_quantitative] min-entail 0.00
- seed: the indianspolis 500 was held before the british grand prix
- rec : the indians is the top of the 500 was

**D43** [ae_drift_procedural] min-entail 0.01
- seed: In this case, the Direct PLUS Loan Request allows some further concessions which I can go through with you if you would like?
- rec : In this case, the Direct Loan Request allows for you to go through with you.

**D44** [ae_drift_quantitative] min-entail 0.00
- seed: Allen Steen is an American martial artist who practices taekwondo and has also been a teacher to some notable taekwondo martial artist.
- rec : Allen Steen is an American martial artist and martial arts martial artist.

**D45** [ae_drift_procedural] min-entail 0.03
- seed: I can help you with that. A co-signer is the spouse of an applicant who initiated an Income-Driven Repayment Plan Request.
- rec : I can help you with that. A co-signer is the person who is who initiated an Income-Driven-Repayo-Rec.

**D46** [ae_drift_scientific] min-entail 0.24
- seed: We perform hydrodynamic simulations of the evolution of the circumstellar medium around a 60 Msol star, from the main sequence through the LBV and Wolf-Rayet stages, up to core collapse.
- rec : We perform hydrodynamic simulations of the evolution of the LBV and L-Ray-Rayet stages, up to core and the L-Ray-Rayet stages, up to the core core.

**D47** [ae_drift_quantitative] min-entail 0.04
- seed: The aerobic gymnastics at the 2009 Asian Indoor Games, hosted by a Vietnamese city, featured four events which were aced by athletes from China who took home three out of the four gold medals.
- rec : The aerobic gymnastics at the Asian Indoor Games, which took the Asian Indoor Games, hosted by a Chinese Olympics, hosted by a Chinese, featured four out of the four gold medals, and three of the gold medals.

**D48** [ae_drift_scientific] min-entail 0.01
- seed: The aim of this paper is to propose a generic discrete-event random simulation model, called VOODB, in order to evaluate the performances of OODBs in general, and the performances of optimization methods like clustering in particular.
- rec : The aim of this paper is to evaluate the performance of the OODBs in general, and the performances in a generic discrete-event simulation.

**D49** [ae_drift_scientific] min-entail 0.01
- seed: We study the nonlinear elastic response of a two-dimensional material to a localized boundary force, with the particular goal of understanding the differences observed between isotropic granular materials and those with hexagonal anisotropy.
- rec : We study the nonlinear-line elastics of the nonline theore-

**D50** [ae_drift_quantitative] min-entail 0.03
- seed: rajendra krishan is the lyricist of the song that madan mohan kohli was musical director for
- rec : raji , the lyrik the song that the song that was musical director for the song was musical-

## Borderline paraphrases (25 lowest min-entailment)

**B1** [ae_para_procedural] min-entail 0.34
- seed: Args: qubit (int): qubit is the qubit measured.
- rec : Args: qubit ( (int)) qubit ()) (int):

**B2** [ae_para_procedural] min-entail 0.35
- seed: In file numpy/core/arrayprint.py, replace: ```python def _recursive_fmt(param, index, indent, curr_width): """ Helper function for _formatArray, to recursively print array elements.
- rec : In file numpypy/core/array/arrayprint.py/coreprint.python, in the name: "".py/recursive//#/array-py//s__/recursals/array/array-

**B3** [ae_para_scientific] min-entail 0.35
- seed: In this note we classify all the spherical nilpotent G-orbits in the Lie algebra of G.
- rec : In this note we classify all the spherical G-nil G-or-nil-or-orbits-orbits in the alge of G.

**B4** [ae_para_procedural] min-entail 0.35
- seed: import textwrap from typing import Any, List, Tuple, Type, Union, cast from sphinx.ext.autodoc import ClassDocumenter, DataDocumenter, ObjectMembers import dagster._check as check from dagster import BoolSource, Field, IntSource, StringSource from dagster._annotations import is_public ) from dagster._core.definitions.configurable import Configurabl
- rec : import textwrap from typing from typing import from typing import textw.ex.com, or.extd ort.import.import textwt.auto import_docdoc importdoc

**B5** [ae_para_procedural] min-entail 0.35
- seed: In file python/ray/autoscaler/updater.py, replace: ```python final_cmd = self.kubectl + [ "exec", "-it" if allocate_tty else "-i", ] + with_interactive(cmd) ``` with: ```python final_cmd = self.kubectl + ["exec", "-it"] final_cmd += [ ] final_cmd += with_interactive(cmd) ``` In file python/ray/autoscaler/updater.py, replace: ```python allocate_tty=
- rec : In file python/python/ray/ray/autoscaler/update/updater.py, filed/python, with python/resolution.py, filed, with with: ```

**B6** [ae_para_procedural] min-entail 0.36
- seed: def run(self, template: str = None, **format_kwargs: Any) -> str: # type: ignore """ Args: - template (str, optional): the template string to format; if not provided, `self.template` will be used - **format_kwargs (optional): keyword arguments to use for formatting Returns: - str: the formatted string """ if template is None: template = self.templa
- rec : def run(self, template: self, template: str = None, template: No, self.temp..temp.template: self.templa

**B7** [ae_para_procedural] min-entail 0.36
- seed: Read the DMV brochures , Let the Buyer Be Aware, and Q&A About Your Title Certificate
- rec : Reads, the D. You're Aware, and Q&A the DMV brochures , Let the Buyer Be Aware, and Q&A About Your Title Title.

**B8** [ae_para_procedural] min-entail 0.36
- seed: addresses = tuple(a for af in address_families for a in af.addressables.keys() if a.target_name == spec.name and not exclude_address(a)) if not addresses: if len(address_families) == 1: _raise_did_you_mean(address_families[0], spec.name) ``` with: ```python def all_included_addresses(): return (a for af in address_families for a in af.addressables.
- rec : addresses = tuple(address_address_families) for a.address_address(address(a))

**B9** [ae_para_procedural] min-entail 0.36
- seed: It will be used to build the /p:Configuration= parameter of MSBuild.
- rec : It will be used to build the /p:Configuration=_the_name of MSBuild.

**B10** [ae_para_procedural] min-entail 0.36
- seed: Returns: MeasureChannel: The Qubit measurement stimulus line """ measure_channels_map = getattr(self, "channels_map", {}).get("measure", {}) qubits = (qubit,) if qubits in measure_channels_map: return measure_channels_map[qubits][0] return None def acquire_channel(self, qubit: int): """Return the acquisition channel for the given qubit.
- rec : Returns: MeasureChannel: The Qubit measurement.

**B11** [ae_para_procedural] min-entail 0.36
- seed: _GH3076: https://github.com/pydata/pandas/issues/3076 pandas 0.10.1 append/put operation (Of course you can simply read in the data and create a new table!) Iterator ~~~~~~~~ Starting in 0.11, you can pass, ``iterator=True`` or ``chunksize=number_in_a_chunk`` to ``select`` and ``select_as_multiple`` to return an iterator on the results.
- rec : _GHGH3076: https://gits.com/pydata.com/i-s-t-s/py/py/p/py/py/spy/py-ss--ss/py-it/s/spy-spy-s--spy-s--s--select--as_multiple-select-as_select_as_multiple--select-it---select-as-

**B12** [ae_para_procedural] min-entail 0.36
- seed: # see https://github.com/numpy/numpy/issues/19624 werror = '-Werror-implicit-function-declaration -Werror' ```
- rec : # see https://github.githubby https://gitpy/numpy/numpy/numpy/numpy/numpy/issues/numpy/issues/19624/ `/n` https://github.github.com/github.com/issue/ https://git.com/numpy/numpy/issues/issue/issues/19624##(#--num-werror---function-log--log---

**B13** [ae_para_procedural] min-entail 0.36
- seed: from IPython.config.configurable import Configurable from IPython.core import oinspect from IPython.core.error import UsageError from IPython.core.inputsplitter import ESC_MAGIC, ESC_MAGIC2 from IPython.external.decorator import decorator from IPython.utils.ipstruct import Struct from IPython.utils.process import arg_split magic_kinds = ('line', 'c
- rec : from IPython.config.config.config.configur import Configurable from IPython.config.config.config.config.autm import Struct from IPython.core.core.core.config.import oinspect from IPython.core.core.error import arg_s_t_in_import_import_import_import_import__main_ import '''s_import', import_import_import', 'c' import from

**B14** [ae_para_procedural] min-entail 0.36
- seed: This second part is, in turn, rendered necessary by the first part, which deals with the application of doctrine and the foundation of self-knowledge.
- rec : This second part is the second, rendered by the first part, which deals with the application of doctrine and the foundation of self-knowledge.

**B15** [ae_para_procedural] min-entail 0.36
- seed: The problem is then formalized as a "Weighted Set Cover" (WSC) problem, for which an approximation algorithm is used.
- rec : The problem is then formalized as a "Weighteded Set Cover" (Weighted) problem, for which an approximation is used, for the use of an approximation algorithm is used.

**B16** [ae_para_scientific] min-entail 0.36
- seed: In this way the renormalization problem for composite lattice operators is avoided altogether, and the connection to the DA is done using perturbation theory in the continuum.
- rec : In this way the renormalization problem for composite lattice lattices is avoided altogether, and the connection to the connection to the DA is done using perturbation theory in the continuum.

**B17** [ae_para_scientific] min-entail 0.37
- seed: Two-photon states generated via optical spontaneous parametric downconversion are shown to be entangled in the parity of their one-dimensional transverse spatial profile.
- rec : Two-photon states generated via optical parametricalvectoronal opticals, and a pair of two-dimensionality, and a few are encared in the parity of the parity of their one-dimensional and a-dimensional spatial-in-time.

**B18** [ae_para_scientific] min-entail 0.37
- seed: A gauge theory of gravity based on a nonlinear realization (NLR) of the local Conform-Affine (CA) group of symmetry transformations is presented.
- rec : A gauge a time, A non-certain gravity of a gravitytry of theorem of a nonlinear-A graviry theory of a nonline,, and

**B19** [ae_para_scientific] min-entail 0.37
- seed: The polarization was measured as a function of the Lambda longitudinal momentum fraction xF and transverse momentum pt.
- rec : The polarization was measured was measured as a function of the Lambda longitudinal momentum fraction xF and transverse momentum fraction was measured.

**B20** [ae_para_procedural] min-entail 0.37
- seed: Returns: MeasureChannel: The Qubit measurement stimulus line """ measure_channels_map = getattr(self, "channels_map", {}).get("measure", {}) qubits = (qubit,) if qubits in measure_channels_map: return measure_channels_map[qubits][0] return None def acquire_channel(self, qubit: int): """Return the acquisition channel for the given qubit.
- rec : Returns: MeasureChannel: The Qubit: The Qubit: MeasureChannels_Map: The Qubit.

**B21** [ae_para_procedural] min-entail 0.37
- seed: deleted file mode 100644 deleted file mode 100644
- rec : 100.b.file.com.ch file.com. 100.b.file..file.filefilefile.new.com.o.

**B22** [ae_para_procedural] min-entail 0.37
- seed: In file qiskit/transpiler/passes/synthesis/unitary_synthesis.py, replace: ```python from qiskit.circuit import ControlFlowOp, Gate ``` with: ```python from qiskit.circuit import ControlFlowOp, Gate, Parameter ``` In file qiskit/transpiler/passes/synthesis/unitary_synthesis.py, replace: ```python available_2q_basis[key] = op ``` with: ```python # 2q
- rec : In file qiskit/transpiler/transpiler/transpiler/transpiler/transpiles/passes/transpile/transpiler/transpiler/passes/transpiler/passes/transpiler/synthesis/synthesis/transpiler_transpiler/transpiler/transpiler_transpiler/passes/trans_synthesis/transpiler_trans_synthesis_synsis.py,

**B23** [ae_para_scientific] min-entail 0.37
- seed: Large samples of high-redshift supernovae (SNe) are potentially powerful probes of cosmic star formation, metal enrichment, and SN physics.
- rec : Large samples of high-relationships of supernovae (e) are potentially powerful probes of cosmic star formation, metal enrichment, and SN physics.

**B24** [ae_para_scientific] min-entail 0.38
- seed: The period map for cubic fourfolds takes values in a locally symmetric variety of orthogonal type of dimension 20.
- rec : The period map for the cubic four-b-s of cubic four-s and cubic four-folds, and cubics and cubic fourfolds takes values in a locally symmetric variety of orthogonalal variety of dimension 20. dimension 20. cubic-s-of-sight-the-seven-the-a-a-s-a-of-the-s-of-s-a-s-the-s-eight-and

**B25** [ae_para_scientific] min-entail 0.38
- seed: Hard-scattering in p-p collisions was discovered in 1972 at the CERN-ISR, the first hadron collider.
- rec : Hard-scattering in p-p collisions was discovered in 1972 at the CERN-IS, the first hadron collider.
