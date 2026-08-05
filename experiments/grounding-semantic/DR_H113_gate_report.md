# DR-H113 TYPED-SWAP kill-gate report

| op | n | judge agree | regrounding | fluent | subtle (of agreed) | still-entailed veto would kill |
|---|---|---|---|---|---|---|
| number | 215 | 0.991 | 0.009 | 0.916 | 0.221 | 0.005 | SURVIVES
| date | 215 | 0.991 | 0.074 | 0.935 | 0.197 | 0.009 | KILLED
| unit | 215 | 0.995 | 0.009 | 0.972 | 0.014 | 0.005 | SURVIVES
| entity | 215 | 0.902 | 0.214 | 0.921 | 0.052 | 0.010 | KILLED
| hedge | 215 | 0.502 | 0.084 | 0.986 | 0.824 | 0.593 | KILLED
| comparative | 215 | 0.972 | 0.014 | 0.953 | 0.081 | 0.024 | SURVIVES
| negation | 215 | 0.963 | 0.019 | 1.000 | 0.000 | 0.005 | SURVIVES

pooled judge agreement: 0.902 (PASS bar >=0.75, hypothesis-kill <0.60)
pooled regrounding: 0.060 (bar <=0.05)
fluency pass: 0.955 (bar >=0.90)
subtle share of agreed: 0.153 (bar >=0.30)
STILL-ENTAILED VETO measurement: 0.056 of judge-certified true negatives would be executed by nli_fwd>=0.8 (campaign-wide number)
surviving operator types: ['number', 'unit', 'comparative', 'negation'] (4/7; hypothesis killed if <3)

## Example swaps per operator (5 each, for the main-session eyeball)

### number
- old→new: `30` → `28` | delta=number-change sev=subtle supported=no agreed=True nli_fwd=0.08
  - seed : Mount Manaia is a dominant landmark approximately 30 kilometres southeast of Whangarei city, the northernmost city in New Zealand, and stands 420 meters.
  - claim: Mount Manaia is a dominant landmark approximately 28 kilometres southeast of Whangarei city, the northernmost city in New Zealand, and stands 420 meters.
- old→new: `2,500` → `1,690` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Trzebież, located in Gmina Police, County Police, West Pomeranian Voivodeship, Poland, has total population of 2,500.
  - claim: Trzebież, located in Gmina Police, County Police, West Pomeranian Voivodeship, Poland, has total population of 1,690.
- old→new: `81` → `71` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : portland was the home team when the score was 81 - 80
  - claim: portland was the home team when the score was 71 - 80
- old→new: `2,000` → `1,721` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.08
  - seed : The Coverdell Education Savings Account allows up to $2,000 a year to be put aside for a student’s education expenses.
  - claim: The Coverdell Education Savings Account allows up to $1,721 a year to be put aside for a student’s education expenses.
- old→new: `1,220` → `1,631` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : if you are working in 2019 and your earnings average more than $1,220 a month you are not disabled.
  - claim: if you are working in 2019 and your earnings average more than $1,631 a month you are not disabled.

### date
- old→new: `2008` → `2009` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : australia had two players who placed during the 2008 us open
  - claim: australia had two players who placed during the 2009 us open
- old→new: `1990` → `1993` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : On 11 March 1990, a year before the formal dissolution of the Soviet Union, Lithuania achieved this status through the Act of the Re-Establishment of the State of Lithuania.
  - claim: On 11 March 1993, a year before the formal dissolution of the Soviet Union, Lithuania achieved this status through the Act of the Re-Establishment of the State of Lithuania.
- old→new: `September` → `February` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : 1940 Saint Louis Billikens football team, under head coach Dukes Duford, lost against Missouri on September 28 with a score of 26–40.
  - claim: 1940 Saint Louis Billikens football team, under head coach Dukes Duford, lost against Missouri on February 28 with a score of 26–40.
- old→new: `2006` → `2004` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : "When a Monster is Born" is a 2006 children's book written by a British author and illustrated by Nick Sharratt in the United Kingdom.
  - claim: "When a Monster is Born" is a 2004 children's book written by a British author and illustrated by Nick Sharratt in the United Kingdom.
- old→new: `1995` → `1997` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Jonny Jakobsen (born 17 November 1963) is a former Bubblegum dance/eurodance singer from Sweden who began his career as a faux-country/pop singer called Johnny Moonshine and releas
  - claim: Jonny Jakobsen (born 17 November 1963) is a former Bubblegum dance/eurodance singer from Sweden who began his career as a faux-country/pop singer called Johnny Moonshine and releas

### unit
- old→new: `years` → `weeks` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : If he was discharged more than 62 years ago, then the National Archives opens all of those records to the public and ordering a copy will be easy.
  - claim: If he was discharged more than 62 weeks ago, then the National Archives opens all of those records to the public and ordering a copy will be easy.
- old→new: `years` → `days` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Katharine McPhee has been performing since she was 21 years old.
  - claim: Katharine McPhee has been performing since she was 21 days old.
- old→new: `years` → `months` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : Kansas has been a band for more than 40 years total.
  - claim: Kansas has been a band for more than 40 months total.
- old→new: `years` → `hours` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Jordan Peele married Chelsea Peretti 14 years after he began his career.
  - claim: Jordan Peele married Chelsea Peretti 14 hours after he began his career.
- old→new: `days` → `weeks` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : If you do not have one of these specific documents or you can not get a replacement for one of them within 10 days
  - claim: If you do not have one of these specific documents or you can not get a replacement for one of them within 10 weeks

### entity
- old→new: `DMV` → `Custom Plates Unit` | delta=entity-swap sev=obvious supported=yes agreed=False nli_fwd=0.06
  - seed : The replacement or transfer personalized plates is not done at the DMV.
  - claim: The replacement or transfer personalized plates is not done at the Custom Plates Unit.
- old→new: `rodrigo ruas` → `marcus vinicios` | delta=entity-swap sev=obvious supported=yes agreed=False nli_fwd=0.00
  - seed : rodrigo ruas has competed in canada , the united states and brazil
  - claim: marcus vinicios has competed in canada , the united states and brazil
- old→new: `U.S.` → `DMV` | delta=entity-swap sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : The southern resident orcas were added to the U.S.
  - claim: The southern resident orcas were added to the DMV
- old→new: `Theophilus` → `Chrysostom` | delta=entity-swap sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Specifically, Theophilus had marched against the monks with soldiers and armed servants, burned their dwellings, and ill-treated those he captured.
  - claim: Specifically, Chrysostom had marched against the monks with soldiers and armed servants, burned their dwellings, and ill-treated those he captured.
- old→new: `New York` → `Constantinople` | delta=entity-swap sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Yes, a lien recorded on an out-of-state title will automatically be printed on a New York title certificate unless your New York title application includes proof that the loan has 
  - claim: Yes, a lien recorded on an out-of-state title will automatically be printed on a New York title certificate unless your Constantinople title application includes proof that the loa

### hedge
- old→new: `about` → `` | delta=omission sev=subtle supported=no agreed=True nli_fwd=0.99
  - seed : Yes, I can send you to another page for information about county use taxes.
  - claim: Yes, I can send you to another page for information county use taxes.
- old→new: `around` → `` | delta=omission sev=subtle supported=no agreed=True nli_fwd=0.99
  - seed : Several applications are given, including a definite improvement of the unclouding problem of [PP1], the prescription of heights of geodesic lines in a finite volume such M, or of 
  - claim: Several applications are given, including a definite improvement of the unclouding problem of [PP1], the prescription of heights of geodesic lines in a finite volume such M, or of 
- old→new: `may` → `will` | delta=none sev=none supported=yes agreed=False nli_fwd=0.97
  - seed : You may be eligible for VA benefits or compensation for surviving spouses if you meet certain requirements. You will also need to provide evidence with your claim showing that one 
  - claim: You may be eligible for VA benefits or compensation for surviving spouses if you meet certain requirements. You will also need to provide evidence with your claim showing that one 
- old→new: `around` → `` | delta=omission sev=subtle supported=no agreed=True nli_fwd=0.98
  - seed : **Building context vectors**: Vectors are extracted by identifying words that appear around the term to be translated within a window of N words, often using association measures l
  - claim: **Building context vectors**: Vectors are extracted by identifying words that appear the term to be translated within a window of N words, often using association measures like mut
- old→new: `suggest` → `prove` | delta=hedge-deletion sev=subtle supported=no agreed=True nli_fwd=0.05
  - seed : We suggest to combine the Anthropic Principle with Many-Worlds Interpretation of Quantum Theory.
  - claim: We prove to combine the Anthropic Principle with Many-Worlds Interpretation of Quantum Theory.

### comparative
- old→new: `most` → `least` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : To begin with, most of the grants that are awarded go to students that have a financial need
  - claim: To begin with, least of the grants that are awarded go to students that have a financial need
- old→new: `first` → `last` | delta=other-factual sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : We first characterize stationary menisci and their breakdown at the coating transition.
  - claim: We last characterize stationary menisci and their breakdown at the coating transition.
- old→new: `more` → `less` | delta=number-change sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : IRVO aims at modeling the interaction between one or more users and the Mixed Reality system by representing explicitly the objects and tools involved and their relationship.
  - claim: IRVO aims at modeling the interaction between one or less users and the Mixed Reality system by representing explicitly the objects and tools involved and their relationship.
- old→new: `after` → `before` | delta=other-factual sev=obvious supported=no agreed=True nli_fwd=0.01
  - seed : Additionally, researchers determined that this visual examination of the bladder wall after stretching is not specific for IC.
  - claim: Additionally, researchers determined that this visual examination of the bladder wall before stretching is not specific for IC.
- old→new: `best` → `worst` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Just take into consideration that it is a long term investment and plan carefully so you can find the school and get the funding options that best work for your current situation.
  - claim: Just take into consideration that it is a long term investment and plan carefully so you can find the school and get the funding options that worst work for your current situation.

### negation
- old→new: `` → `not` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : :raises ValueError: if :attr:`user_project` is set.
  - claim: :raises ValueError: if :attr:`user_project` is not set.
- old→new: `` → `not` | delta=none sev=none supported=yes agreed=False nli_fwd=0.05
  - seed : are you wondering why there are duplicates of some schools?
  - claim: are not you wondering why there are duplicates of some schools?
- old→new: `` → `not` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : Thus just a few zealots can prevent consensus or even the formation of a robust majority.
  - claim: Thus just a few zealots can not prevent consensus or even the formation of a robust majority.
- old→new: `` → `not` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : The latter were found using methods from logic and the paper continues a case study in the general program of extracting effective data from prima-facie ineffective proofs in the f
  - claim: The latter were not found using methods from logic and the paper continues a case study in the general program of extracting effective data from prima-facie ineffective proofs in t
- old→new: `` → `not` | delta=negation sev=obvious supported=no agreed=True nli_fwd=0.00
  - seed : A new class of space time codes with high performance is presented.
  - claim: A new class of space time codes with high performance is not presented.