# DR-2 pilot judge eyeball - 50 certified negatives, stratified

Judge Qwen/Qwen3-32B-FP8, 50387 judged, 22838 certified negatives.
Grade: a pair PASSES if the claim is genuinely unsupported vs the seed (the corruption changed factual content). Bar: >= 85% pass.

**E1** [H112-sent] delta=entity-swap sev=subtle nli_fwd=0.397
- seed : We generalize our analysis to models with anisotropic interactions, showing that, as long as the lattice is correctly embedded in the plane, such discretely holomorphic parafermions exist for particular values of the couplings which we identify as the anisotropic FZ points.
- claim: We generalize our analysis to models with anisotropic interactions, showing that, as long as the lattice is correctly embedded in the plane, such discretely holomorphic parafermions exist for particular values of the couplings which we identify as the anisotropic boundary points.
- changed: anisotropic boundary points

**E2** [H112-sent] delta=entity-swap sev=subtle nli_fwd=0.735
- seed : This paper explores the experimentally relevant range of measurement strategies between the two, where the rate of inconclusive results is minimized for a bounded-error rate.
- claim: This paper explores the experimentally relevant range of measurement strategies between the aforemention, where the rate of inconclusive results is minimized for a bounded-error rate.
- changed: aforemention

**E3** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.005
- seed : Paul Gillard, a British TV actor, took a policeman role in the "Crane" and Captain Fraser in the "Sergeant Cork."
- claim: Paul Gillard, a British TV actor, took a policeman role in the "Crane" and Captain America: The Winter in the "Sergeant Cork."
- changed: Captain America: The Winter

**E4** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.012
- seed : the purpose of the frequency used to service the area of katherin is national
- claim: the purpose of the frequency used to service the area of the United States is national
- changed: the United States

**E5** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.008
- seed : It is shown that all macroscopic morphological characters (the habit plane, the macroshear and the orientational relationship) are expressed through elastic moduluses Cij of an initial bcc phase.
- claim: It is shown that all macroscopic morphological characters (the habit plane, the macroshear and the orientational relationship) are expressed through elastic moduluses Cij of an initial and an intermediate phase phase.
- changed: an initial and an intermediate phase

**E6** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.082
- seed : LGBT rights in Oregon received 70% support with an 854 sample size and 72% of 1,006 sample size.
- claim: LGBT rights in Oregon received the highest level of support with an 854 sample size and 72% of 1,006 sample size.
- changed: the highest level of support

**E7** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.005
- seed : The Spring of 1984 is when Katharine McPhee was born.
- claim: The Spring of 1984 is when my first was born.
- changed: my first

**E8** [H112-sent] delta=number-change sev=obvious nli_fwd=0.003
- seed : Kactoos (launched in 2009) is a defunct social network service and group-buying site created by Kactoos Group and headquartered in Miami, Florida, which needs no registration to join.
- claim: Kactoos (launched in February 2017 ) is) is a defunct social network service and group-buying site created by Kactoos Group and headquartered in Miami, Florida, which needs no registration to join.
- changed: February 2017

**E9** [H112-sent] delta=number-change sev=obvious nli_fwd=0.001
- seed : The first round race of the 1983 Australian Touring Car Championship (a CAMS sanctioned motor racing title) was held at Calder Park Raceway in Melbourne, Victoria on February 6, and was won by Allan Moffat.
- claim: The third round race of round race of the 1983 Australian Touring Car Championship (a CAMS sanctioned motor racing title) was held at Calder Park Raceway in Melbourne, Victoria on February 6, and was won by Allan Moffat.
- changed: third round race

**E10** [H112-sent] delta=number-change sev=obvious nli_fwd=0.627
- seed : Are you older than 21 but not yet 24, unaccompanied and either homeless or self-supporting and at risk of being homeless?
- claim: Are you older than 21 but not yet 21 years of age, unaccompanied and either homeless or self-supporting and at risk of being homeless?
- changed: older than 21 but not yet 21 years of age

**E11** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.001
- seed : 25% of all artificial radiation belts came from kazakhstan
- claim: The most expensive of all artificial radiation belts came from kazakhstan
- changed: The most expensive of all artificial radiation belts

**E12** [H112-sent] delta=number-change sev=obvious nli_fwd=0.000
- seed : Fluctuations in the spontaneous beating activity of isolated cardiac cells were studied over a timescale of six decades.
- claim: Fluctuations in the spontaneous beating activity of isolated cardiac cells were studied over a timescale of 24 hours. The ..
- changed: 24 hours

**E13** [H112-sent] delta=omission sev=subtle nli_fwd=0.196
- seed : It introduced a new lexicon of approximately fifteen terms to oppose evolution without using religious language, changing over one hundred uses of "creation" to "intelligent design" in its 1987 drafts.
- claim: It introduced a new lexicon of approximately fifteen terms to oppose evolution without using religious language, changing over one hundred uses of "creation" to "intelligent design" in its first drafts.
- changed: first drafts

**E14** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.019
- seed : 86.40% of Bali's population adheres to Balinese Hinduism.
- claim: 86.40% of India 's population's population adheres to Balinese Hinduism.
- changed: India

**E15** [H112-sent] delta=number-change sev=obvious nli_fwd=0.002
- seed : 1.575 canadian viewers (million) watched the episode that james hurst & shelley scarrow wrote
- claim: In the last 24 canadian viewers (million) watched the episode that james hurst & shelley scarrow wrote
- changed: last 24 canadian viewers (million)

**E16** [H112-sent] delta=omission sev=obvious nli_fwd=0.502
- seed : The S-stars have been suggested to be a brief transitional phase as stars evolve from oxygen-rich M-type stars into carbon stars, through the dredge up of carbon from He-shell burning.
- claim: The S-stars have been suggested to be a brief transitional phase as stars evolve from oxygen-rich M-type stars into carbon stars, through the dredge up of the shell andshell burning.
- changed: the dredge up of the shell andshell burning

**E17** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.677
- seed : The phase diagram determined in the Hubbard interaction versus temperature plane shows novel reentrant behavior in the Mott transition due to the competition between Fermi-liquid formation and magnetic correlations under geometrical frustration.
- claim: The phase diagram determined in the Hubbard interaction versus temperature plane shows novel reentrant behavior in the Mott transition due to the competition between the two directions of-liquid formation and magnetic correlations under geometrical frustration.
- changed: the two directions of-liquid formation

**E18** [H112-sent] delta=number-change sev=obvious nli_fwd=0.355
- seed : Even if you hold a permit or license issued by another state, as long as you are under 16 you cannot drive in NY State.
- claim: Even if you hold a permit or license issued by another state, as long as you are under 18 years of age you cannot drive in NY State.
- changed: under 18 years of age

**E19** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.013
- seed : We confirm the detection of an absorption line plausibly identified as OVIII Ly-alpha from the warm-hot intergalactic medium associated with a small group of galaxies along the line of sight, as originally reported by Fang et al.
- claim: We confirm the detection of an absorption line plausibly identified as OVIII Ly-alpha from the warm-hot intergalactic medium associated with a small group of galaxies along the line of sight, as originally reported by the physicists at the.
- changed: the physicists at the

**E20** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.001
- seed : You need to call the DMV Revenue Accounting Unit
- claim: You need to call the police.
- changed: police

**E21** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.226
- seed : Do you need other VA benefits and services as well?
- claim: Do you need other health benefits and benefits and services as well?
- changed: health benefits and benefits

**E22** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.002
- seed : Championnats Internationaux de France de Tennis is a French sports events.
- claim: Championnats Internationaux de France de Tennis is a list of the world sports events.
- changed: a list of the world sports events

**E23** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.427
- seed : Most information can not be updated because it must be accurate as of the day you originally signed your FAFSA form.
- claim: Most information can not be updated because it must be accurate as of the day you originally signed your registration form.
- changed: registration form

**E24** [H112-sent] delta=number-change sev=subtle nli_fwd=0.637
- seed : Kozjak pri Ceršaku is a settlement of 165 people on a 2.2 km (0.8 sq mi) area in Šentilj, Drava, Slovenia.
- claim: Kozjak pri Ceršaku is a settlement of 165 people on a 2.2 km (1 mi ) area in Šenti) area in Šentilj, Drava, Slovenia.
- changed: 1 mi

**E25** [H112-sent] delta=entity-swap sev=obvious nli_fwd=0.787
- seed : Returns: pandas.DataFrame: if pandas is None: raise ImportError("pandas is required to create a DataFrame") if dtypes is None: dtypes = {} avro_schema, column_names = _avro_schema(read_session) frames = [] for block in self: dataframe = _to_dataframe_with_dtypes( _avro_rows(block, avro_schema), column_names, dtypes ) frames.append(dataframe) return
- claim: Returns: pandas.DataFrame: if pandas is None: raise ImportError("pandas is required to create a DataFrame") if dtypes is required to create a Data: dtypes = {} avro_schema, column_names = _avro_schema(read_session) frames = [] for block in self: dataframe = _to_dataframe_with_dtypes( _avro_rows(block, avro_schema), column_names, dtypes ) frames.append(dataframe) return
- changed: if dtypes is required to create a Data

**E26** [H112-sent] delta=number-change sev=subtle nli_fwd=0.002
- seed : Edgewise, which aired on Saturday evenings on MSNBC from 1996-1997, was hosted by John Hockenberry.
- claim: Edgewise, which aired on Saturday evenings on MSNBC from 9 to 11 p.m, was hosted by John Hockenberry.
- changed: from 9 to 11 p.m

**E27** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.005
- seed : guest is the notes for both the pre - 2013 mandarin chinese program on ctv
- claim: guest is the notes for both the english and mandarin mandarin chinese program on ctv
- changed: english and mandarin mandarin chinese

**E28** [H112-sent] delta=omission sev=obvious nli_fwd=0.782
- seed : For an even Dirichlet character psi, we obtain a formula for L(1,psi) in terms of a sum of Dirichlet L-series evaluated at s=2 and s=3 and a rapidly convergent numerical series involving the central binomial coefficients.
- claim: For an even Dirichlet character psi, we obtain a formula for L(1,psi) in terms of a sum of the following formulas: evaluated at s=2 and s=3 and a rapidly convergent numerical series involving the central binomial coefficients.
- changed: the following formulas: evaluated at s=2 and s=3

**E29** [H112-sent] delta=omission sev=subtle nli_fwd=0.009
- seed : The Python runtime version string is 3.12.9, as shown in the first part of the output on line 1.
- claim: The Python runtime version string is 3.12.9, as shown in the following table: part of the output on line 1.
- changed: as shown in the following table: part of the output on line 1

**E30** [H112-sent] delta=other-factual sev=subtle nli_fwd=0.019
- seed : Jenga was designed by Leslie Scott to be used by at least one player.
- claim: Jenga was designed by Leslie Scott to be used by any kind of player.
- changed: any kind of player

**E31** [H112-sent] delta=other-factual sev=obvious nli_fwd=0.147
- seed : To investigate the intrinsic radiation property of BL Lac objects, we estimated the Doppler factor with the VLA or MERLIN core and the total 408 MHz luminosity for a sample of 170 BL Lac objects.
- claim: To investigate the intrinsic radiation property of BL Lac objects, we estimated the Doppler factor with the VLA or MERLIN core and the total sensitivity of the MHz luminosity for a sample of 170 BL Lac objects.
- changed: total sensitivity of the MHz luminosity

**E32** [H112-sent] delta=number-change sev=obvious nli_fwd=0.001
- seed : Iliyan Garov(born 8 January 1984) is a football centre back defender from Plovdiv.
- claim: Iliyan Garov(born 8 May 1974 ) is a) is a football centre back defender from Plovdiv.
- changed: 8 May 1974

**E33** [H112-sent] delta=omission sev=obvious nli_fwd=0.002
- seed : thiago alves has played two times in são paulo , brazil and manta , ecuador
- claim: thiago alves has played in the major league times in são paulo , brazil and manta , ecuador
- changed: two

**E34** [H112-sent] delta=entity-swap sev=subtle nli_fwd=0.648
- seed : According to the provided evidence, this separation offers benefits such as better separation of concerns, the ability to use pluggable platform-specific UI frameworks (like Compose or SwiftUI), and the ability to test business logic code with pure multiplatform unit tests.
- claim: According to the provided evidence, this separation offers benefits such as better separation of concerns, the ability to use pluggable platform-specific test frameworks (like Compose or SwiftUI), and the ability to test business logic code with pure multiplatform unit tests.
- changed: test frameworks

**E35** [H112-long] delta=number-change sev=obvious nli_fwd=0.220
- seed : These detections have changed our understanding of planet formation ``beyond the snowline'' by demonstrating that Neptune-mass planets with separations of several AU are common. 
- claim: These detections have changed our understanding of planet formation ``beyond the snowline'' by demonstrating that Neptune-mass planets with separations of several degrees of magnitud are common. 
- changed: degrees of magnitud

**E36** [H112-long] delta=entity-swap sev=subtle nli_fwd=0.612
- seed : you 'll need a Premium DS Logon account . Your My HealtheVet or ID.me credentials won t work on the eBenefits website . Go to eBenefits to sign in , register , or upgrade your DS Logon account to Premium . **not masked**), - 0 for pixels that are padding (i.e. Coronation Street (colloquially known as Corrie) is a British television soap opera created by Granada Television and written by Tony Warre
- claim: you 'll need a Premium DS Logon account . Your My HealtheVet or ID.me credentials won t work on the eBenefits website . Go to eBenefits to sign in , register , or upgrade your DS Logon account to Premium . **not masked**), - 0 for pixels that are padding (i.e. Coronation Street (colloquially known as Corrie) is a British television soap opera created by Tony and written by Tony Warren, based on hi
- changed: created by Tony

**E37** [H112-long] delta=number-change sev=obvious nli_fwd=0.027
- seed : (ii) two qubits are brought together to realize a gate, and 
- claim: (ii) When all the qu qubits are brought together to realize a gate, and 
- changed: When all the qu qubits

**E38** [H112-long] delta=number-change sev=obvious nli_fwd=0.053
- seed : seven of the geological naming of venus occurred in 1997 corinthians was in the 1st position with a difference of 5 the 29th season episode was titled beaver says good - bye which aired on april 16 , 1959 Our models DCGCN(single) and DCGCN(ensemble)consist of full GCN layers, removing the burden of employing a recurrent encoder to extract non-local contextual information in the bottom layers. 
- claim: seven of the geological naming of venus occurred in 1997 corinthians was in the 1st position with a difference of 5 the 29th season episode was titled beaver says good - bye which aired on air on the 19th and 20th Our models DCGCN(single) and DCGCN(ensemble)consist of full GCN layers, removing the burden of employing a recurrent encoder to extract non-local contextual information in the bottom lay
- changed: air on air on the 19th and 20th

**E39** [H112-long] delta=entity-swap sev=obvious nli_fwd=0.104
- seed : none of the judges appointed by coolidge obtained a senior status all states except two had a preliminary average of at least 8 sweden won the nordic skiing olympics 2 consecutive times televendita was shown on la sorgente sat 3 Nelia Penman, a British Liberal Party politician and barrister, received the least number of votes in the Sevenoaks division at the 1964 general elections. 
- claim: none of the judges appointed by the Supreme Court obtained a obtained a senior status all states except two had a preliminary average of at least 8 sweden won the nordic skiing olympics 2 consecutive times televendita was shown on la sorgente sat 3 Nelia Penman, a British Liberal Party politician and barrister, received the least number of votes in the Sevenoaks division at the 1964 general electi
- changed: the Supreme Court

**E40** [H112-long] delta=other-factual sev=subtle nli_fwd=0.023
- seed : manuel reynante covered 2100 km more than that of paquito rivas Battle of Carmona is one of Scipio's first major battles in Spain and occurred in 207 BC. 
- claim: manuel reynante covered 2100 km more than that of paquito rivas Battle of Carmona is one of Scipio's most important and major major battles in Spain and occurred in 207 BC. 
- changed: most important and major major battles

**E41** [H114-sent] delta=omission sev=subtle nli_fwd=0.158
- seed : In file mesonbuild/modules/gnome.py, add: ```python # The `install_header` argument will be used by mkenums() when # not using template files, so we need to forcibly unset it # when generating the C source file, otherwise we will end up # installing it c_file_kwargs['install_header'] = False ```
- claim: In file mesonbuild/modules/gnome.py, add: ```python # The `install_header` argument will be used by mkenums() when # install using template files, so we need to forcibly unset it # when generating the C source file, otherwise we will end up # installing it c_file_kwargs['install_header'] = False ```
- changed: install using template files

**E42** [H114-sent] delta=negation sev=obvious nli_fwd=0.003
- seed : I argue that the prediction of physical discreteness at the Planck scale in loop gravity is a reasonable conclusion that derives from a sensible ensemble of hypotheses, in spite of some contrary arguments considered in an interesting recent paper by Dittrich and Thiemann.
- claim: I am not that the prediction of physical discreteness at the Planck scale in loop gravity is a reasonable conclusion that derives from a sensible ensemble of hypotheses, in spite of some contrary arguments considered in an interesting recent paper by Dittrich and Thiemann.
- changed: I am not that

**E43** [H114-sent] delta=entity-swap sev=obvious nli_fwd=0.086
- seed : Ptolemy XIV is a one of the co-ruler of the cleopatra VII philopator ruler
- claim: The first person to be na is a one of the co-ruler of the cleopatra VII philopator ruler
- changed: The first person to be na

**E44** [H114-sent] delta=entity-swap sev=obvious nli_fwd=0.773
- seed : Indira Gandhi was born in British India .
- claim: The following are was born in British India .
- changed: The following are

**E45** [H114-sent] delta=other-factual sev=obvious nli_fwd=0.004
- seed : In this work, we apply black box methods (methods not requiring input) to find excited-state energies.
- claim: In this work, we apply black box methods (methods s requiring input) to find excited-state energies.
- changed: s requiring input

**E46** [H114-sent] delta=number-change sev=obvious nli_fwd=0.003
- seed : The minimum rustc version required to install gping via Cargo is 1.67.0 or greater.
- claim: The minimum rustc version required to install gping via Cargo is 3.5 or greater.
- changed: 3.5

**E47** [H114-sent] delta=omission sev=obvious nli_fwd=0.002
- seed : This typically means either pytest is not installed in the active conda environment or the wrong Python executable is being invoked for test execution.
- claim: This typically means either pytest is install installed in the active conda environment or the wrong Python executable is being invoked for test execution.
- changed: install installed

**E48** [H114-sent] delta=omission sev=subtle nli_fwd=0.014
- seed : Of course, no problem with that. You can always consider to use an income-driven repayment as a more healthy alternative, so you don't have to delay your payments.
- claim: Of course, you problem with that. You can always consider to use an income-driven repayment as a more healthy alternative, so you don't have to delay your payments.
- changed: you

**E49** [H114-sent] delta=number-change sev=obvious nli_fwd=0.005
- seed : Jack Nicklaus of United States placed 1st with the score of 69-69-70-73=281 in the final round of 1971 PGA Championship, which was played on February 25–28, 1971 at the Palm Beach Gardens, Florida.
- claim: Jack Nicklaus of United States placed 1st with the score of 18,239, in the final round of 1971 in the final round of 1971 PGA Championship, which was played on February 25–28, 1971 at the Palm Beach Gardens, Florida.
- changed: 18,239

**E50** [H114-sent] delta=entity-swap sev=obvious nli_fwd=0.146
- seed : "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1}, ...
- claim: "B1980" : {0 : 3.2, 3.2 : 1.3, 2 : .1}, ...
- changed: 3.2
