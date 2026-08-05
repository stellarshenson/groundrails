# R10-H111 stage 0b - eyeball sample (degeneracy-gated referee)

Model facebook/mbart-large-50, best_p 0.2. Thresholds: nll <= 5.710, distinct3 >= 0.952, maxrun <= 1.0 (all from p=0.05 recons).
Adjudication bar: < 1 in 10 of the ADMITTED DRIFT pairs below is a meaning-preserving
paraphrase (and degenerate repetition should now be absent - flag any that slipped).

## Admitted drift at p=0.2 (50 random)

**D1** [quantitative] min-entail 0.00
- seed: Chris Pratt began acting when they were 21 years old.
- rec : Chris Pratt began acting when they were only a few years ago.

**D2** [scientific] min-entail 0.01
- seed: We investigate the Coulomb excitation of low-lying states of unstable nuclei in intermediate energy collisions ($E_{lab}\sim10-500$ MeV/nucleon).
- rec : We investigate the Coulomb excitation of low-lying states of low-lyinging states of unstable nucle-in and the effects of the phen i.e. in--itmc-mediate-s-t-s---s of-a-s---s-----s--a--s-----s--e-a-s-a---s---s--s-

**D3** [quantitative] min-entail 0.07
- seed: Dyscritothamnus is a genus of Mexican flowering plants in the Asteraceae family of the order Asterales.
- rec : Dyscritothamnus is a genus of Mexican flowering plants in the Asteraceae family of the order.

**D4** [procedural] min-entail 0.00
- seed: * **Integration with Probabilistic and Data-Driven Methods:** The approach can be integrated with Partially Observable Markov Decision Processes (POMDPs) to handle n-best hypotheses for estimating belief state distributions.
- rec : * **Inteur with Probabilistic and Data-Driven Methods:** The approach can be integrated with Partially Observation of Probabilistic and Data-Driven...

**D5** [procedural] min-entail 0.00
- seed: Gladly. First of all, you should review the Child Disability Starter Kit. This kit answers common questions about applying for Supplemental Security Income (SSI) benefits for children.
- rec : Gladly. First of all of first of the World of the first in the place,

**D6** [quantitative] min-entail 0.01
- seed: Nadia Cassini changed her name at some point in her life.
- rec : Nadia's name at some point in her life.

**D7** [quantitative] min-entail 0.01
- seed: Coweb, a 2009 Hong Kong martial arts film, was written by Chan Wing-Sun and directed by Xiong Xin Xin.
- rec : Co Co, a 2009 Hong Kong martial arts film, was written by Chan Wing-Sun and directed by Xiong-Sun.

**D8** [quantitative] min-entail 0.01
- seed: Danny Elfman was associated with Oingo Boingo during his active years.
- rec : Danny Elf. Danny Elfman was the site of the most popular.

**D9** [scientific] min-entail 0.02
- seed: As a result, we establish the robustness and the usefulness of the multiple matter-wave solitons in the spinor BECs.
- rec : As a result, we have no doubtlessly found the

**D10** [scientific] min-entail 0.02
- seed: The magnetization measured at $T$ = 2 K exhibited two metamagnetic transitions at $H_{\rm m1}$ = 31 kOe and $H_{\rm m2}$ = 44.7 kOe, for $H \parallel$ [100] with a saturation magnetization of 1.6 $\mu_{\rm B}$/Ce.
- rec : The magnet-magnetization of $H \r_{\_{{\rm${{\rm2 \rm_{{{s \r_{{{{{{}}_{}{}}}$$$$$

**D11** [quantitative] min-entail 0.00
- seed: Danedream is a famous racehorse trained by Peter Schiergen.
- rec : Daneam is a famous racehorse trained by Peter Schiergen.

**D12** [quantitative] min-entail 0.01
- seed: The 1958 Holy Cross Crusaders football team was an American football team that represented the College of the Holy Cross as an independent during the 1958 NCAA University Division football season that opened their season with a 0-17 loss to Pittsburgh, but won their next 5 games.
- rec : The 1958 , the 1958 Holy Cross--Clas-of-Closed-Ace-A-Cy-A-A--Cycoonly-a-hoursethire-Aligu--A-C-A-American-A-Churgee-A-A-A-A-C-C-A-C-C-A-A-C-A-A-A-e-C-A-C-C-

**D13** [quantitative] min-entail 0.01
- seed: Susan Kohner was involved with three awards between 1958 and 1962 one of which was the Golden Globe Award.
- rec : Susan Kohner was in the Golden Globe.

**D14** [scientific] min-entail 0.01
- seed: The results show that the Ostwald rule, which predicts which phase will nucleate, must be modified probabilistically when the new phases are almost equally stable.
- rec : The results show that the Ostwald rule, which predicts is the only to predicted which, which predicts which predicts, which, which predictions, which is not stablely, is not only, must be, and then,

**D15** [procedural] min-entail 0.02
- seed: if a lens user leaves your organization you can not log in to lens. does this apply?
- rec : if a lens user is not for your company.

**D16** [quantitative] min-entail 0.00
- seed: Four members of the Citadel Bulldogs wrestling team have earned All-American honors, including Dan Thompson and Odie Delaney.
- rec : Four members of the Citadel Bulldogs wrestling team have earned All-A.

**D17** [procedural] min-entail 0.08
- seed: In file datastore/google/cloud/datastore/query.py, replace: ```python query_pb2.QueryResultBatch.NO_MORE_RESULTS, ``` with: ```python _NO_MORE_RESULTS = query_pb2.QueryResultBatch.NO_MORE_RESULTS _NO_MORE_RESULTS, ``` In file datastore/google/cloud/datastore/query.py, replace: ```python if response_pb.batch.end_cursor == b'': # Empty-value for byte
- rec : In file datastore/google/cloud/google/cloud/datastore/datastore/google/datastore/google/datastore/google/datastore/google/cloud/datastore/datastore/query/google/cloud/datastore/google/google/datastore/google/datastore/google/datastore/datastore/google/google/google/datastore/datastore/google/google/google/datastore/google/google/datastore/datastore/google/

**D18** [procedural] min-entail 0.00
- seed: In that case the NYS Department of Motor Vehicles, "DMV" may send you a Notice of Registration Suspension. The notice will advise you that if you pay all of the tolls, fees or other charges owed to the tolling authority, or have them dismissed or transferred, the suspension will not take effect.
- rec : In that case the NYS Department of Motor Vehicles, "DMV" may be dismissed, and will be dismissed.

**D19** [procedural] min-entail 0.10
- seed: input context length) model = GPT(model_config) ```
- rec : input = (in-t) = GPT(model) model(model) = GPT(model) model) ``(model)``(``(````=

**D20** [scientific] min-entail 0.02
- seed: Our light curve analysis leads to the radius of the secondary, $R_{\rm B} = 0.167 \pm 0.006$ \rsun, and the semimajor axis of the orbit, $a = 7.54 \pm 0.30 \rsun = 0.0351 \pm 0.0014$ AU.
- rec : Our light curve, $a = \rm = \rm \m = 0.006 = 0.001$ 0.0005$ AU.

**D21** [procedural] min-entail 0.00
- seed: According to the evidence, factors influencing gambling tendencies in casinos include **sound**, **odour**, and **lighting**.
- rec : According to the evidence, factors influencing gambling tendencies, factors in casinos, and **sounds, and in casinos.

**D22** [scientific] min-entail 0.00
- seed: This suggests that we may make a distinction between different chaotic behaviours of the orbit via the gravitational waves.
- rec : This suggests that we may have.

**D23** [scientific] min-entail 0.01
- seed: As another application of these techniques, we prove that every countable group $C$ can be realized as a group of outer automorphisms of a group $N$, where $N$ is a finitely generated group having Kazhdan's property (T) and containing exactly two conjugacy classes.
- rec : As another application of these techniques, we can prove that the following is true.

**D24** [procedural] min-entail 0.01
- seed: We will let you know once you complete your application.
- rec : We will let you know once you complete your complete.

**D25** [scientific] min-entail 0.03
- seed: We show that addition of flavors to these theories (via additional non-compact branes) leads to local meta-stable supersymmetry breaking minima, closely related to those of SQCD with massive flavors.
- rec : We show that addition of flavors to these theories (via additional non-to-comparable-to-the-the-se-chance-a-a-way-to-the-the-s--in-the-a-way-to-the-the-e-s-w-to-the-a-the-be-added-to-a-the-the-s-of-the-way-s-the-the-a-the-

**D26** [scientific] min-entail 0.00
- seed: The analysis based on the mean-field approximation indicates that the observed patterns result from the presence of Hopf and wave bifurcations in the considered system.
- rec : The analysis of the observed patterns.

**D27** [scientific] min-entail 0.09
- seed: Given an orientable weakly self-dual manifold X of rank two, we build a geometric realization of the Lie algebra sl(6,C) as a naturally defined algebra L of endomorphisms of the space of differential forms of X.
- rec : Given an orientable weakness of the positions of the algebray of the ,C.

**D28** [scientific] min-entail 0.45
- seed: A relatively modest investment in a ground-based network of small ($\sim 0.5 {\rm m}$ telescopes could provide the needed coverage and so dramatically increase the effectiveness of transit timing observations.
- rec : A relatively modest investment in a ground-based network of small ($\sim }m} {\m} m} m}$ telescopes, could provide the needed coverage and so dramatically increase the effectiveness of transit timing observations..

**D29** [procedural] min-entail 0.01
- seed: The `BooleanFieldListFilter` class is defined in `django/contrib/admin/filters.py` at line 36, where it inherits from `BooleanOnlyFieldListFilter` and implements a `__init__` method to set up the filter with specific lookup parameters.
- rec : The `__init__init__` class defined in `Boo__in`, `__init` class.

**D30** [quantitative] min-entail 0.00
- seed: Catalonia, rather than being a single city, is a Spanish region consisting of 4 separate provinces.
- rec : Catalonia, rather than a single city, is a single city.

**D31** [procedural] min-entail 0.30
- seed: def __init__(self, exists): self._exists = exists def __eq__(self, other): if not isinstance(other, self.__class__): return NotImplemented return self._exists == other._exists def modify_write(self, write_pb, **unused_kwargs): """Modify a ``Write`` protobuf based on the state of this write option.
- rec : def __e __init__(self, self._existsself, other, other._exists = __in_exists)== other: __e_exists._exists = exists def __e__(self, other): if not isinstance(self, other): if not isinstance(other, self.__class__): return NotImplemented return-a-rearthing-__(self._existscore__)

**D32** [scientific] min-entail 0.00
- seed: It can be done by using a quantum semantics arising from the deep logical structure of quantum theory.
- rec : It can be done by using a quantum semantics.

**D33** [procedural] min-entail 0.01
- seed: Excerpt 1 details the **Swiss Alpine Club** yearbooks, noting that the articles cover themes such as **mountaineering**, club activities, and sports, but it does not explicitly list "skiers" or describe the members with that specific combination of terms.
- rec : Excerpt the "" is a list of the **Swiss Alpine Club** yearbooks, noting the yearbook, the

**D34** [scientific] min-entail 0.22
- seed: All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.
- rec : All next-to--to-next-to-next-to-next-to-next-the-order-to-neether-to-ne-the-the-to-next-to-netheme of-to-next-to-the-to-the-to-the--to-the-to-to-the--to-the-a-record-to-next-to-the-to-the-

**D35** [quantitative] min-entail 0.02
- seed: The total assets of Mowi ASA are over 20 billion more than the total equity.
- rec : The total assets of Mowi ASA is estimated in the value of the market value of Mowi ASA are over 20 billion more than the total equity.

**D36** [quantitative] min-entail 0.04
- seed: Guitarist Adam Nowak (who was born September 28, 1963 in Poznan, Poland) currently lives near Torun with his family, including four children, Stanislaw, Jan, Barbara and Zofia.
- rec : In the video, a new and a hard-she, Adam Nowak, Adam Nowak (who was born September 28, 1963 in Poznan in 1963 in Poznan, Poland) currently lives near Torun with his family, including four children, Stanislaw, Jan, Barbara, and Zofia.

**D37** [scientific] min-entail 0.01
- seed: We find that at the one-loop level this model may accommodate the electroweak phase transitions that are strongly first-order in a reasonably large region of the parameter space.
- rec : We find that at the one-

**D38** [procedural] min-entail 0.04
- seed: :type include_syntax: bool :param include_syntax: (Optional) Flag to enable syntax analysis of the current document.
- rec : :type include_syntax_syntax: bool: bool :type_syntaxtax: bool:

**D39** [quantitative] min-entail 0.01
- seed: The Serbs were the majority ethnic group in Yugoslavia from 1953-1981 the Hungarians had the least.
- rec : The Serb and the Serbs, the Serbs and the most of the Serbs were the first.

**D40** [quantitative] min-entail 0.10
- seed: Maynila sa mga Kuko ng Liwanag Directed by Lino Brocka
- rec : Maynila sa mga may-hang-angangot na ng Li-neng-yong-muk-ang Mayo-ong-yon----inhers--s-in-in--ong---the---the-them

**D41** [quantitative] min-entail 0.00
- seed: Ong Kian Ming, a Malaysian politician in the DAP party, was elected to Bangi, Selangor in 2018.
- rec : Mayo, the government to the Household, in 2018.

**D42** [scientific] min-entail 0.00
- seed: Our study is based on high-frequency recordings of the S&P500, DAX and WIG20 indices over the interval May 2004 - May 2006.
- rec : Our study is based on-frequentage and is on high-time-frequency-on-freque-on-time-time-frequenncy-time-frequency-the--freque-the-on-on-freque-frequency--or--freque--frequeque the study-freque-the-freque-frefrequent-freque-freque-fre-freque-frequen

**D43** [scientific] min-entail 0.24
- seed: We show that for a monotonic distribution over an alphabet of size $k$, each probability parameter costs essentially $0.5 \log (n/k^3)$ bits, where $n$ is the coded sequence length, as long as $k = o(n^{1/3})$.
- rec : We show that for a monotonic distribution over an \log (n/k^3)$ (n/k/k^3)$) $0.5 (n (n/$) $k$ (n/$) $n/ $$$) $($$) $

**D44** [procedural] min-entail 0.00
- seed: Sure, you can. To do it, please bring a check or money payable to for at least the minimum amount on your statement, to your local DMV office.
- rec : Sure, you can. To do it, please bring a check or your cash.

**D45** [scientific] min-entail 0.31
- seed: As applications, we show how to use the trace to show that the diagram representation is faithful, and to compute leading coefficients of certain Kazhdan--Lusztig polynomials.
- rec : applications, we show how to use the to show to that the diagram representation is faithful, and to compute, and to compute leading coefficients of certain Kazhdan--Luhdh----Luh---Lu-Luh.

**D46** [quantitative] min-entail 0.00
- seed: Kherkhedi is a village in India with 80 households, a total population of 407(2011): 210 female and 197 male and effective literacy rate of 70.31%.
- rec : Kherkhedi is a village with a population of India with a village: Khererkhedidi district

**D47** [scientific] min-entail 0.03
- seed: We show that the globular cluster mass function (GCMF) in the Milky Way depends on cluster half-mass density (rho_h) in the sense that the turnover mass M_TO increases with rho_h while the width of the GCMF decreases.
- rec : The GCMF function is a non-dewealine, multi--sectionalthy-stack-wide-scaled-to--a-s-slow-height-s-a-to--

**D48** [procedural] min-entail 0.06
- seed: Personalized plates are standard series plates that have a combination of numbers and letters that you select.
- rec : Personalized plates are standard series plates that have a combination of numbers and numbers that you select.

**D49** [procedural] min-entail 0.01
- seed: The implementation of `GenericForeignKey.get_prefetch_queryset` is found in lines 173-191, where it groups instances by content type ID to optimize database queries.
- rec : The implementation of the following:

**D50** [scientific] min-entail 0.00
- seed: As a corollary, it is shown that the formation of a black hole with an S**(n-2) x S**1 horizon from that with an S**(n-1) horizon must be non-axisymmetric in asymptotically flat space-times.
- rec : to the formation of the S**(n-2) horizon from that with an S**(n-2) horizon.

## Borderline paraphrase (25 lowest min-entailment among admitted paraphrases)

**B1** [procedural] min-entail 0.36
- seed: >>> df.loc['viper'] max_speed 4 shield 5 Name: viper, dtype: int64 List of labels.
- rec : >>> df.loc['['viper'] max_speed_speed 4 , max_speed 4 ]]></a_s

**B2** [scientific] min-entail 0.39
- seed: This paper considers the propagation of shallow-water solitary and nonlinear periodic waves over a gradual slope with bottom friction in the framework of a variable-coefficient Korteweg-de Vries equation.
- rec : This paper--coe--a--in-coe--infra--de Vries-athere-a-water--a--co-a--s-line-in---re--a--in-e-to----a--a--

**B3** [procedural] min-entail 0.40
- seed: Sortedness of the result is not guaranteed Parameters ---------- other : Index or array-like Returns ------- intersection : Index """ if not isinstance(other, RangeIndex): return super(RangeIndex, self).intersection(other) if not len(self) or not len(other): return RangeIndex._simple_new(None) first = self[::-1] if self._step < 0 else self second =
- rec : Sortedness of the results of the result is not guaranteed

**B4** [procedural] min-entail 0.45
- seed: In file src/transformers/trainer.py, replace: ```python self.save_model(output_dir) ``` with: ```python self.save_model(output_dir, _internal_call=True) ``` In file src/transformers/trainer.py, replace: ```python self.save_model(output_dir) ``` with: ```python self.save_model(output_dir, _internal_call=True) ``` In file src/transformers/trainer.py,
- rec : In file src/transformers/transformers/transformers/trainer.py, replace: ```python self.save_model(output_model(output_model)_model))) ```)` In file src/transformers/transform

**B5** [procedural] min-entail 0.46
- seed: basis_gates (list[str]): List of basis gate names to unroll to.
- rec : basis_gates (e_gatess_gates_gates (list[a]]): gt basis_gates_gates names to unroll to.

**B6** [procedural] min-entail 0.48
- seed: and to the end new_codes = codes.copy() pos = len(codes) - n_nans new_codes[0:pos] = codes[~na_mask] new_codes[pos:] = -1 codes = new_codes self._codes = codes return return self._constructor(values=codes, dtype=self.dtype, ``` with: ```python sorted_idx = nargsort(self, ascending=ascending, na_position=na_position) self._codes = self._codes[sorted
- rec : and to the end new_codes = codes.copy() - new_codes[pos: n_pos = codes.__rerenew_codes(pos()=_._``_codes._pos_codes.__codes._s_pos._codes =-pos._codes._pos=codespos._pos.-_codes._pos =_codes[pos=_pos=_

**B7** [procedural] min-entail 0.49
- seed: In file mysql/check.py, replace: ```python slave_io_running = self._collect_string('Slave_IO_Running', results) slave_sql_running = self._collect_string('Slave_SQL_Running', results) slave_io_running = (slave_io_running.lower().strip() == "yes") slave_sql_running = (slave_sql_running.lower().strip() == "yes") ``` with: ```python slave_io_running = 
- rec : In file mysql/py..py._Sli_in file_sg____s________s_sql____s__s_____s_ss________s_sql____________________________s____________

**B8** [scientific] min-entail 0.49
- seed: We find a huge difference in the XMLD contrast between the two types of magnetic domains, which we discuss in terms of intrinsic magneto-crystalline anisotropy of XMLD of the Co layer.
- rec : We find a huge difference in, and we find the difference in the XMLD--crystalline aniso-cry-s of the XMLD of the Co-crystalline of the Co layer.

**B9** [procedural] min-entail 0.50
- seed: ``` with: ```python def assert_numpy_array_equal(np_array, assert_equal, strict_nan=False, err_msg=None): """Checks that 'np_array' is equivalent to 'assert_equal'.
- rec : ``` with: ```py'pypy_ as assert_numpy_array_array_array_e_equal(np_array_array_array_array_equal, assert_n_name(n__na=, assert_array_array_array_array_array_array_array_equal_equal, assert_e_equal_assert_name__arra

**B10** [procedural] min-entail 0.50
- seed: Yes, Claude HUD is a Claude Code plugin.
- rec : Yes, Claude HUD, is a Claude, a Claude Coded, a Claude Codes a Claude Code.

**B11** [quantitative] min-entail 0.52
- seed: Lance White (born August 31, 1946) is a Canadian former municipal and provincial level politician and engineer, who went to the University of Alberta and served on the Edmonton city council from 1983 until 1992.
- rec : Lance White (born August 31, 1946) is a Canadian former municipal and provincial level politician and engineer, politician and engineer, who went to the University of Alberta and the University of the Alberta and served on the Edmonton city council from 1983 until 1992.

**B12** [procedural] min-entail 0.52
- seed: Yes, fuel-core is a Fuel client implementation, as explicitly stated in the provided evidence.
- rec : Yes, fuel-core is a client is a Fuel-core, and the identification of a client-based ae-to-fuel-core-client, client implementation, as explicitly stated in the provided evidence.

**B13** [procedural] min-entail 0.53
- seed: It is uncommon for a program to be larger or smaller than that range.
- rec : It is unrealy of the unused. It is unmon for a program to be larger or less than that.

**B14** [scientific] min-entail 0.54
- seed: As a result we find a sufficient condition for an open 7-manifold to admit a closed 3-form of $\tilde G_2$-type.
- rec : As a result we find a sufficient condition for an open 7-manifold to admit a closed 3-form of $\tilde G_2 G_2_2$-type..

**B15** [scientific] min-entail 0.54
- seed: Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC).
- rec : Predictions are shown for distributions for diphoton pairs and for distributions of diphoton pairs produced at the energy of the Largerronron of the Large Hadron Collider (L/c).

**B16** [quantitative] min-entail 0.55
- seed: There are more vineyards than there are wineries in Condrieu AOC
- rec : There are more than a lot of vineyards than there are winess in Condridri A A.

**B17** [scientific] min-entail 0.58
- seed: However, for natural choices of parameters, backreaction from the Kaluza-Klein gravitons may well become important.
- rec : However, for natural choices of parameters of parameters, and the natural choices of parameters, backreaction from the Kaluza-K-Klein gravitons, may well become important.

**B18** [procedural] min-entail 0.60
- seed: The signs and symptoms of hypoglycemia resolve after blood glucose levels have returned to normal.
- rec : Thes and symptoms and the signs of hypoglycycymia after blood glucose levels have returned to normal.

**B19** [procedural] min-entail 0.61
- seed: Examples:: # Let's see how to increase the vocabulary of Bert model and tokenizer tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h') model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h') num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2']) print('We have added', num_added_toks, 'tokens') # 
- rec : Examples:: # Let's see how to increase the vocabulary of the model and tokenizer token.add_tokenizer.add_tokens = Wav2Vec2CTCTCTCtc.from_pretr_pretrn'=='''<'new_tokenizer.'') model = Bert_added_added_tokenizer = Wav2Vec2CTCTCTokenizer.from_pretr

**B20** [scientific] min-entail 0.61
- seed: One of the key results of our present investigation is a great deal of simplification in the geometrical understanding of the nilpotent (anti-)BRST symmetry invariance.
- rec : One of the key results of our present investigation is a great deal of simplification of the geometrical understanding of the nil--) and the nil--BRST symmetry invari.

**B21** [procedural] min-entail 0.61
- seed: This is only used for recording the log filenames in Redis.
- rec : This is only used for recording the filenames in Red log logiss in Red logis.

**B22** [scientific] min-entail 0.61
- seed: We find that the nucleation rate is suppressed at early times even after global variables such as the magnetization and energy have apparently reached their time independent values.
- rec : We find that the nucless and nucleation is in the nucleation is suppressed at early rate even after global variables such as the magnetization and energy have apparently reached their time independent values.

**B23** [scientific] min-entail 0.61
- seed: We suggest that the simple insertion of a short DNA fragment into the gene may suffice to turn an unknotted into a knotted structure in this protein.
- rec : We suggest that the simple insertion of a short DNA into the gene may suffice to turn an unknotted into a knotted structure in this protein.

**B24** [scientific] min-entail 0.62
- seed: Our main result in this paper is the following: Given $H^m, H^n$ hyperbolic spaces of dimensional $m$ and $n$ corresponding, and given a Holder function $f=(s^1,...,f^{n-1}):\partial H^m\to \partial H^n$ between geometric boundaries of $H^m$ and $H^n$.
- rec : Our main result in this paper is the following:

**B25** [procedural] min-entail 0.63
- seed: Index and columns labels may be non-numeric, e.g.
- rec : Index and columns labels may be non-num-numer-num-numericic, e.
