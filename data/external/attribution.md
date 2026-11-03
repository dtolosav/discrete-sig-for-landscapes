# Data and software attribution

The landscapes and sequence classes come from Benjamin et al.,
[*Homology of homologous knotted proteins*](https://doi.org/10.1098/rsif.2022.0727).
We use [source commit 6262a2b](https://github.com/katherine-benjamin/ph-knotted-proteins/tree/6262a2bf760155a9a3aad2672d1e51f07a01dfe9),
specifically `data/validation/landscapes/interpolated_x1` and `interpolated_x6`.
The upstream MIT notice is preserved in `LICENSE-Benjamin.txt`.

Each degree-one cohort has 513 chains: 363 in nine named classes and 150 labelled
Other. The experiments exclude Other. `input_checksums.json` records every
landscape checksum and the corrected classification-file checksum.

`trefoil_list.csv` contains the 513 logical source rows. We removed 79 malformed
duplicate rows from the earlier local copy and corrected the C-tail of `5jes_B`
from 4 to 5, consistent with its stored depth. No protein was added by this repair.

FRUITS is included as an MIT-licensed submodule pinned to
`fd5bd3c7e4e55f6de5bcc9ff1791a64685a85c56`; its licence remains in that submodule.
Chen signatures use [pySigLib](https://github.com/datasig-ac-uk/pySigLib).
These packages retain their own authorship and licensing.
