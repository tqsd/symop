# `symop_proto` Overview

The library consists of the following modules:
- `symop_proto/core` - Core implementations with very little logic. `core` doesn't import anything from outside.
- `symop_proto/envelopes` - Envelope implementations, currently only Gaussian. Imports only from `core/`
- `symop_proto/labels` - Label logic, only imports from the `core`.
- `symop_proto/algebra` - Algebra
