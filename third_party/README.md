# External GLCM dependency

The historical extraction invokes `GLCM_Features.m`, whose header identifies Avinash Uppuluri and a modification date of 20 November 2008. The recovered file's SHA256 is:

`e858a08f28c3c377968207f273f13b10af727318898f305bf7d309067c31620b`

The local working tree contains a copy under `artifacts/third_party/GLCM_Features.m`. It is excluded from Git. Its redistribution terms were not established during the audit; this external code is not covered by the project's MIT license. Obtain/document the original dependency and its terms before a public release. MATLAB wrappers load this local artifact directory explicitly. Do not substitute another implementation without checking feature definitions against the frozen workbooks.
