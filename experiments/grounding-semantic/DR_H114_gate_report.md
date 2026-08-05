# DR-H114 XATTN-BLIND kill-gate

```json
{
 "n": 800,
 "recon_rate": 0.0163,
 "degen_rate": 0.1425,
 "drift_rate": 0.8413,
 "nli_fwd_mean": 0.7147,
 "nli_fwd_ge08_share": 0.6478,
 "post_veto_yield": 0.2963,
 "per_type": {
  "entity": {
   "n": 209,
   "recon": 0.01,
   "degen": 0.177,
   "drift": 0.813
  },
  "hedge": {
   "n": 86,
   "recon": 0.012,
   "degen": 0.047,
   "drift": 0.942
  },
  "negation": {
   "n": 59,
   "recon": 0.0,
   "degen": 0.102,
   "drift": 0.898
  },
  "number_date": {
   "n": 190,
   "recon": 0.011,
   "degen": 0.232,
   "drift": 0.758
  },
  "positional": {
   "n": 23,
   "recon": 0.0,
   "degen": 0.13,
   "drift": 0.87
  },
  "relverb": {
   "n": 233,
   "recon": 0.034,
   "degen": 0.086,
   "drift": 0.88
  }
 },
 "verdict": "SURVIVES-to-pilot",
 "bars": "KILL if recon >= 0.60 or degen > 0.30"
}
```

## Eyeball - 20 blinded decodes across locus types

**X1** [hedge] span="can" nli_fwd=0.99
- seed : The provided evidence section is empty, so I cannot identify the specific code block or line numbers in `django/db/models/query.py` where `returning_fields` are removed during `update_conflicts` handl
- blind: The provided evidence section is empty, so I donnot identify the specific code block or line numbers in `django/db/models/query.py` where `returning_fields` are removed during `update_conflicts` handl
- dec  : don

**X2** [negation] span="not" nli_fwd=0.01
- seed : personalized plates that give the appearance of an offical plate, combinations that are considere obscene, combinations that do not have one letter
- blind: personalized plates that give the appearance of an offical plate, combinations that are considere obscene, combinations that do have have one letter
- dec  : have

**X3** [number_date] span="2008" nli_fwd=0.98
- seed : After Ma Ying-jeou's landslide victory in the 2008 presidential election, the Kuomintang (KMT) reclaimed the presidency, ending the period in which it had lost power.
- blind: After Ma Ying-jeou's landslide victory in the presidenti presidential election, the Kuomintang (KMT) reclaimed the presidency, ending the period in which it had lost power.
- dec  : presidenti

**X4** [relverb] span="proposed" nli_fwd=0.99
- seed : Jets and outflows from young stellar objects are proposed candidates to drive supersonic turbulence in molecular clouds.
- blind: Jets and outflows from young stellar objects are are candidat candidates to drive supersonic turbulence in molecular clouds.
- dec  : are candidat

**X5** [entity] span="Kuomintang" nli_fwd=0.93
- seed : After Ma Ying-jeou's landslide victory in the 2008 presidential election, the Kuomintang (KMT) reclaimed the presidency, ending the period in which it had lost power.
- blind: After Ma Ying-jeou's landslide victory in the 2008 presidential election, the KMT) re (KMT) reclaimed the presidency, ending the period in which it had lost power.
- dec  : KMT) re

**X6** [positional] span="`image_loader.o` and `gaussian_blur.o`." nli_fwd=0.98
- seed : The linker error block is found in the provided evidence starting at line 56, where the build process fails with undefined references to OpenCV functions like `cv::imread` and `cv::imwrite` in `image_
- blind: The linker error block is found in the provided evidence starting at line 56, where the build process fails with undefined references to OpenCV functions like `cv::imread` and `cv::imwrite` in undefin
- dec  : undefined references to OpenCV functions like `cv::imread` and `cv::imwrite`

**X7** [hedge] span="can" nli_fwd=0.99
- seed : For example, if you delete the only You can then verify the changes look ok, then git :ref:`commit <contributing.commit-code>` and :ref:`push <contributing.push-code>`.
- blind: For example, if you delete the only You then then verify the changes look ok, then git :ref:`commit <contributing.commit-code>` and :ref:`push <contributing.push-code>`.
- dec  : then

**X8** [negation] span="no" nli_fwd=0.02
- seed : The novel cancellations necessary for ultraviolet finiteness first appear at one loop in the guise of the "no-triangle hypothesis".
- blind: The novel cancellations necessary for ultraviolet finiteness first appear at one loop in the guise of the ""-triangle hypothesis".
- dec  : "

**X9** [number_date] span="the years" nli_fwd=0.93
- seed : maxxforce 5 has a v6 cylinder layout and the years produced are 2007 - current
- blind: maxxforce 5 has a v6 cylinder layout and the number produced are 2007 - current
- dec  : the number

**X10** [relverb] span="reveal" nli_fwd=0.99
- seed : We investigate distortions in the velocity fields of disc galaxies and their use to reveal the dynamical state of interacting galaxies at different redshift.
- blind: We investigate distortions in the velocity fields of disc galaxies and their use to to the dynamical state of interacting galaxies at different redshift.
- dec  : to

**X11** [entity] span="paul mcgee" nli_fwd=0.04
- seed : paul mcgee scored 6 more goals for rovers than brendan bradley
- blind: The following are the top 5 scored 6 more goals for rovers than brendan bradley
- dec  : The following are the top 5

**X12** [positional] span="method for dictionaries" nli_fwd=0.73
- seed : dict.update : Similar method for dictionaries.
- blind: dict.update : Similar .update : Similar.update : Similar.
- dec  : .update : Similar.update : Similar

**X13** [hedge] span="would" nli_fwd=0.98
- seed : Maybe you would find more attractive the online option, then?
- blind: Maybe you find find more attractive the online option, then?
- dec  : find

**X14** [negation] span="not" nli_fwd=0.00
- seed : Based on the provided evidence, specific version numbers for the integrator modules are not mentioned.
- blind: Based on the provided evidence, specific version numbers for the integrator modules are mentioned mentioned.
- dec  : mentioned

**X15** [number_date] span="one" nli_fwd=0.99
- seed : It aims to collect these conveniences in one centralized library to streamline the usage of RxJava with Kotlin.
- blind: It aims to collect these conveniences in a centralized library to streamline the usage of RxJava with Kotlin.
- dec  : a

**X16** [relverb] span="get" nli_fwd=1.00
- seed : To get a record of all your information, you'll need a lifetime driving record.
- blind: To a a record of all your information, you'll need a lifetime driving record.
- dec  : a

**X17** [entity] span="non-Hermitian" nli_fwd=0.51
- seed : We investigate properties of the most general PT-symmetric non-Hermitian Hamiltonian of cubic order in the annihilation and creation operators as a ten parameter family.
- blind: We investigate properties of the most general PT-symmetric PT-symmetric Hamiltonian of Hamiltonian of cubic order in the annihilation and creation operators as a ten parameter family.
- dec  : PT-symmetric Hamiltonian of

**X18** [positional] span="." nli_fwd=0.96
- seed : Assuming the deleterious mutations in the Penna ageing model to affect mainly the young ages, we get an enhanced mortality at very young age, followed by a minimum of the mortality, and then the usual
- blind: Assuming the deleterious mutations in the Penna ageing model to affect mainly the young ages, we get an enhanced mortality at very young age, followed by a minimum of the mortality, and then the usual
- dec  : .

**X19** [hedge] span="can" nli_fwd=0.99
- seed : For example, if you delete the only You can then verify the changes look ok, then git :ref:`commit <contributing.commit-code>` and :ref:`push <contributing.push-code>`.
- blind: For example, if you delete the only You then then verify the changes look ok, then git :ref:`commit <contributing.commit-code>` and :ref:`push <contributing.push-code>`.
- dec  : then

**X20** [negation] span="no" nli_fwd=0.99
- seed : This benefit will be charged to you based on the training time, no matter how much money you are reimbursed. Part-time training rates reduce your GI Bill benefit by half a month for each month you enr
- blind: This benefit will be charged to you based on the training time, , matter how much money you are reimbursed. Part-time training rates reduce your GI Bill benefit by half a month for each month you enro
- dec  : ,
