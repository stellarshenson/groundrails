# R10-H111 contrastive-judge eyeball (validation)

Judge Qwen/Qwen3-32B-FP8, n=500 judged.

**J1** [ae_drift_procedural] delta=omission sev=obvious
- seed: If you receive other government benefits, such as workers' compensation, public disability benefits, or pension based on work not covered by Social Security.
- rec : If you receive other government benefits, you may also receive a government benefits.
- changed: you may also receive a government benefits

**J2** [ae_drift_procedural] delta=entity-swap sev=obvious
- seed: In file bigquery/google/cloud/bigquery/client.py, add: ```python location str: (Optional) Default location for jobs / datasets / tables.
- rec : In file bigpypypy/google/google/cloud/cloud/bigque/cloud/cloud/cloud/bigpy/bigquery/cloud/cloud/cloud/cloud/cloud/client.py, add:
- changed: bigpypypy/google/google/cloud/cloud/bigque/cloud/cloud/cloud/bigpy/bigquery/cloud/cloud/cloud/cloud/cloud/client.py

**J3** [ae_drift_scientific] delta=omission sev=obvious
- seed: We provide the definition and fundamental properties of algebraic elements with respect to an operator satisfying hypothesis (h).
- rec : We provide the definition and fundamental properties of algebraic elements of algebraic elements with respect to an operator satisfying hypothesis (a) and an operator's satisfying hypothesis (h).
- changed: of algebraic elements with respect to an operator satisfying hypothesis (a) and an operator's satisfying hypothesis (h)

**J4** [ae_drift_procedural] delta=omission sev=obvious
- seed: No relevant code lines were provided in the input, so I cannot identify the specific error block or line numbers in `ReportChart.tsx` where the module not found for `'./chart'` is reported.
- rec : No relevant code lines were provided in the input, so I would have a problem of the module not found for `ReportChart.ts'` is reported.
- changed: I cannot identify the specific error block or line numbers in `ReportChart.tsx` where the module not found for `'./chart'`

**J5** [ae_drift_quantitative] delta=entity-swap sev=obvious
- seed: fano was the group c winner in both the 1978 - 79 and the 1989 - 90 seasons
- rec : The starringed in the winner in both the 1978 - 79 and the 1989 - 90s
- changed: The starringed in the winner

**J6** [ae_drift_procedural] delta=omission sev=obvious
- seed: In that case, you don't need to pay the plate fees or MCTD fee and you could also be able to transfer the registration fees to your new vehicle.
- rec : In that case, you don't need to pay the plate or the plate fees or MC.
- changed: MCTD fee and you could also be able to transfer the registration fees to your new vehicle

**J7** [ae_drift_scientific] delta=omission sev=obvious
- seed: Despite their claimed biological plausibility, most self organizing networks have strict topological constraints and consequently they cannot take into account a wide range of external stimuli.
- rec : Despite their claimed biological plausibility, most self--sense-e, they are not a strictly theoretical.
- changed: self--sense-e, they are not a strictly theoretical

**J8** [ae_drift_procedural] delta=omission sev=obvious
- seed: It is necessary to register your vehicle at the DMV office and not online, over the phone, or by mail.
- rec : It is necessary to register your vehicle at the DMV office, by mail, by mail, by
- changed: by mail, by mail, by

**J9** [ae_para_procedural] delta=other-factual sev=obvious
- seed: As the speed of a planing hull increases, its draft decreases.
- rec : As the speed of the speed of a planing hull increases, its draft decreases.
- changed: the speed of the speed

**J10** [ae_drift_procedural] delta=omission sev=obvious
- seed: Excellent. The video is a great way to learn about federal grants, loans, and work-study jobs.
- rec : Excellent. The video is a great.
- changed: way to learn about federal grants, loans, and work-study jobs

**J11** [ae_drift_procedural] delta=omission sev=obvious
- seed: It may not be the most efficient solution and might need fixing to override more methods.
- rec : It may not be the most efficient..
- changed: might need fixing to override more methods

**J12** [ae_para_procedural] delta=entity-swap sev=subtle
- seed: In that case your GI Bill Statement of Benefits will show you how much of your benefits you've used and how much you have left to use.
- rec : In that case your GI Bill Statement of Bene Benefits will be show how much of your benefits you've used and how much you have left to use.
- changed: Bene Benefits

**J13** [ae_drift_procedural] delta=omission sev=obvious
- seed: The maximum loan amount is the cost of attendance at the school your child will attend minus any other financial assistance he or she receives. Remember that the cost of attendance is determined by the school.
- rec : The maximum loan amount is the cost of attendance at the school your child will attend minus the school.
- changed: minus the school

**J14** [ae_drift_scientific] delta=omission sev=obvious
- seed: We have surveyed a ~0.9-square-degree area of the W3 giant molecular cloud and star-forming region in the 850-micron continuum, using the SCUBA bolometer array on the James Clerk Maxwell Telescope.
- rec : We have surveyed a ~---square-de-square-degree area of the W3 giant.
- changed: a ~---square-de-square-degree area of the W3

**J15** [ae_drift_scientific] delta=omission sev=obvious
- seed: Using first principles, classical potentials, and elasticity theory, we investigated the structure of a semiconductor/semiconductor interface with a high lattice mismatch, SiC/Si(001).
- rec : Using first principles, classical principles, and elastics, and elasticity,
- changed: principles, and elastics, and elasticity

**J16** [ae_drift_scientific] delta=omission sev=obvious
- seed: Various holographic approaches to QCD in five dimensions are explored using input both from the putative non-critical string theory as well as QCD.
- rec : Various holographic approaches to the study of five dimensions are explored using the data from the same as the .
- changed: QCD in five dimensions as well as QCD

**J17** [ae_drift_procedural] delta=entity-swap sev=obvious
- seed: In `src/sync/manager.rs` around lines 108-119, the code attempts to retrieve a connection from `self.pool` via `self.pool.get_connection()`, send data using `conn.send(data)`, and release the connection back to the pool with `self.pool.release(conn)`.
- rec : In `src/sync/src/src/sync/sync/s/manager.php` around lines 108-119, the code attempts to retrieve a connection from `self.pool` via `self.
- changed: src/sync/src/src/sync/sync/s/manager.php

**J18** [ae_drift_procedural] delta=omission sev=obvious
- seed: Would you like to know about appealing to a Veterans Law Judge?
- rec : Would you like to know about a Law?
- changed: a Law

**J19** [ae_drift_scientific] delta=entity-swap sev=obvious
- seed: The clusters (Cl0016+16, MS0451.6-0305, MS1054.4-0321 and Abell 2218) are popular SZ targets and serve as commissioning observations.
- rec : The clusters (Cir-e-98.6-0305, MS0451.6-0305, MS105.6-0305, MS1054.4-0321 and Abell-003 are targets.
- changed: Cir-e-98.6-0305, MS105.6-0305, Abell-003

**J20** [ae_drift_quantitative] delta=omission sev=obvious
- seed: Safee Sali played more matches for Kuala Lumpur than Telekom Melaka and Sarawak which was founded in 1974.
- rec : Safee Sali played more than 10 more matches for the time of the year.
- changed: more than 10 more matches for the time of the year

**J21** [ae_drift_quantitative] delta=omission sev=obvious
- seed: There were 38 Women's handball matches at the XXXI Olympiad.
- rec : There are no longer than 3 matches
- changed: no longer than 3 matches

**J22** [ae_drift_quantitative] delta=other-factual sev=obvious
- seed: Drowning Girl is an oil and polymer piece that Roy Lichtenstein finished in 1963.
- rec : Drowning Girl is an oil and an oil and polymer piece that was finished in 1963.
- changed: an oil and an oil and polymer

**J23** [ae_drift_scientific] delta=omission sev=obvious
- seed: Key results are summarized with emphasis on present science and future prospects.
- rec : Key results are summarized with emphasis on present science and future science and future prospects.
- changed: future science and future prospects

**J24** [ae_para_procedural] delta=omission sev=subtle
- seed: Which type of compression to choose depends on your specific needs and data.
- rec : Which type of compression to choose depends on your needs and data.
- changed: specific

**J25** [ae_drift_quantitative] delta=omission sev=obvious
- seed: 2065 Spicer, discovered by Indiana University's Indiana Asteroid Program on September 9, 1959, was a dark and eccentric asteroid from the middle region of the asteroid belt named after American anthropologist Edward H. Spicer.
- rec : 2065 Spicer was a . Spicer, discovered by Indiana University's Indiana Asteroid Program on September 9, 1959, was a black-headed and an asteroid.
- changed: dark and eccentric asteroid from the middle region of the asteroid belt named after American anthropologist Edward H. Spicer

**J26** [ae_drift_procedural] delta=omission sev=obvious
- seed: The fees on this chart are for commercial vehicles that are less than 18000 lbs in weight. For diesel commercial vehicles that are 8501 pounds or more in weight, there is an additional fee.
- rec : The fees on this chart are for commercial vehicles that are less than 18000 lbs in weight.
- changed: For diesel commercial vehicles that are 8501 pounds or more in weight, there is an additional fee.

**J27** [ae_para_procedural] delta=omission sev=obvious
- seed: I am not sorry to inform you that it is not acceptable
- rec : I am not sorry to inform you that it is not to be
- changed: to be

**J28** [ae_para_scientific] delta=omission sev=obvious
- seed: In this paper based on the notion of entanglement witness, a new measure of entanglement called floating entanglement witness measure is introduced which satisfies some of the usual properties of a good entanglement measure.
- rec : In this paper based on the notion of entanglement witness, a new measure of entanglement called floating entanglement witness measure is introduced which is the same as to be a good measure of the measure of entanglement.
- changed: is the same as to be a good measure of the measure of entanglement

**J29** [ae_para_procedural] delta=other-factual sev=obvious
- seed: No, the component does not create additional wrapper DOM nodes.
- rec : No, the component does not creates additional wrap DOMs.
- changed: creates additional wrap DOMs

**J30** [ae_drift_quantitative] delta=other-factual sev=obvious
- seed: More people voted yes on question two of the 2016 Virginia ballot measures.
- rec : More people voted yes on the day, the most voted on the year's, the top two of the 2016 ballot measures..
- changed: the day, the most voted on the year's, the top two of the 2016 ballot measures

**J31** [ae_drift_scientific] delta=omission sev=obvious
- seed: We develop a mixture-based approach to robust density modeling and outlier detection for experimental multivariate data that includes measurement error information.
- rec : We develop a mixture-based approach to the use-A and the dataset-based approach to robustly model the measurement of measures.
- changed: the use-A and the dataset-based approach to robustly model the measurement of measures

**J32** [ae_drift_procedural] delta=omission sev=obvious
- seed: In the current timeline of the comic, Largo is in a relatively successful relationship with Hayasaka Erika.
- rec : In the current timeline of the comic, the comic is in the same time line of the comics.
- changed: 

**J33** [ae_drift_scientific] delta=omission sev=obvious
- seed: Using the six basic models, we have constructed a multi-expert multi-model called the Super Special Hexagonal Fuzzy and Neutrosophic model.
- rec : Using the six basic models, we have a multi-expert multi-models.
- changed: multi-expert multi-models

**J34** [ae_drift_procedural] delta=omission sev=obvious
- seed: The attempt to establish a permanent settlement at Port Couvreux was abandoned because the experimental sheep farming initiative failed to create a viable economic base, leading to the evacuation of the last inhabitants in 1931.
- rec : The attempt to establish a permanent settlement at Port Couvreux was abandoned in 1931.
- changed: because the experimental sheep farming initiative failed to create a viable economic base, leading to the evacuation of the last inhabitants

**J35** [ae_drift_scientific] delta=omission sev=obvious
- seed: Analyzing such data with probabilisic models can be delicate because the simple exchangeability assumptions underlying many boilerplate models no longer hold.
- rec : Analyzing such data with probabilisic models can be a very useful because of the exchanges.
- changed: a very useful because of the exchanges

**J36** [ae_para_procedural] delta=other-factual sev=obvious
- seed: Lancelot Andrewes was ordained a priest in 1580.
- rec : Lancelot Andrewes was a man of faith, a man and the Holy. Andrewes was ordained a man, was ordained a priest in 1580.
- changed: a man of faith, a man and the Holy. Andrewes was ordained a man,

**J37** [ae_drift_procedural] delta=omission sev=obvious
- seed: It's applies if you can not able to make our payments, but under certain circumstances. For example, are you having financial difficulties?
- rec : It's applies if you to the same as a bridge.
- changed: It's applies if you to the same as a bridge.

**J38** [ae_drift_scientific] delta=entity-swap sev=obvious
- seed: For MgCNi3, the thermopower is negative, - 12.5 mikroV/K, at 300 K.
- rec : For MgCNCNii3, the thermopower is negative, - 12.5, - 12.5 mikroV/ -5 mikroV, - 12.5 mikroV.
- changed: MgCNCNii3

**J39** [ae_drift_quantitative] delta=number-change sev=obvious
- seed: jeff gordon was the only driver with a time of 3:09:45
- rec : jeff gordon was the only driver with a time of 3 3:03:45
- changed: 3 3:03:45

**J40** [ae_drift_procedural] delta=other-factual sev=obvious
- seed: Based on the provided evidence, there is no information regarding "laboratory-controlled concepts" or a measurement of "control compound comprehension." Excerpt 3 explicitly states that "most compounds do not contain a hyphen," contradicting the premise that laboratory-controlled concepts measure control compound comprehension.
- rec : Based on the provided evidence, there is no information regarding "laboratory-control-controlled concepts" or a measured of a "control compound comprehension."
- changed: laboratory-control-controlled concepts, a measured of a

**J41** [ae_drift_scientific] delta=entity-swap sev=obvious
- seed: A fault-tolerant memory architecture based on low-density parity-check codes is proposed and the existence of reliable memories for the adversarial failure model is proved.
- rec : A aim-tolerant memory architecture based on the memory architecture of the adversarial failure model is proposed.
- changed: aim-tolerant

**J42** [ae_para_procedural] delta=omission sev=subtle
- seed: To set up Direct Deposit you will need a bank or credit union account. You will need to have your checkbook or a copy of your bank statement with you.
- rec : To set up Direct Deposit, you will need a bank or credit or bank account. You will need to have your checkbook or a copy of your bank statement with you.
- changed: credit or bank account

**J43** [ae_drift_procedural] delta=omission sev=obvious
- seed: In standard Russian, vowel reduction is not reflected in the orthography, whereas in Belarusian, it is.
- rec : In standard Russian, the word of the language, of the world, Russian, it is not.
- changed: vowel reduction is not reflected in the orthography, whereas in Belarusian, it is

**J44** [ae_para_scientific] delta=omission sev=obvious
- seed: GLAST will be launched at the end of 2007, opening a new and important window on a wide variety of high energy astrophysical phenomena .
- rec : GLAST will be launched at the end of 2007, opening a new and important window on a wide variety of high energy astrophys phy .
- changed: phy

**J45** [ae_drift_scientific] delta=omission sev=obvious
- seed: We show that a class of torsional compactifications of the heterotic string are dual to conventional Kahler heterotic string compactifications.
- rec : We show that a class of torsional compactifications of the heterotic string are dual or double-resi--s.
- changed: dual or double-resi--s

**J46** [ae_drift_scientific] delta=omission sev=obvious
- seed: We let c(T,k) denote the number of transversals of size k in a rooted tree T.
- rec : We let c(T,k) the number of the number of a rows of the t.
- changed: the number of the number of a rows of the t

**J47** [ae_para_procedural] delta=entity-swap sev=obvious
- seed: partition_key_range (Optional[PartitionKeyRange]): Partition key range to execute with.
- rec : partition_key_key_range (Optional): Partition key_ range to execute with.
- changed: partition_key_key_range

**J48** [ae_drift_quantitative] delta=entity-swap sev=obvious
- seed: Invesco Ltd. Number of employees are 8900 (2019)
- rec : In.co Ltd.
- changed: In.co Ltd.

**J49** [ae_para_quantitative] delta=omission sev=obvious
- seed: both of the 2014 fifa world cup qualification games were held in guatemala
- rec : The following of the 2014 FIFA world cup qualification games were held in guatemala
- changed: The following of the 2014 FIFA world cup qualification games

**J50** [ae_drift_procedural] delta=omission sev=obvious
- seed: Good, in that case, you may want to work with an accredited attorney, a claim agent or a Veterans Service Officer. We recommend you these professionals because they are trained and certified in the VA claims and appeals processes.
- rec : Good, in that case, in that case, you may want to work with an accredited attorney, a claim agent or a Veterans Service Officer.
- changed: We recommend you these professionals because they are trained and certified in the VA claims and appeals processes.
