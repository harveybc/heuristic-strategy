# Constant Channel Retirement

Owner approved the narrowly scoped retirement of px.er_1 on 2026-10-24.
No other feature, scaler constant, model width, window or business rule changes.
Recipe revision 3 retains the v2 serialization schema for existing consumers.

Tests before implementation:

1. Parent identity and input pins are required; altered parents fail.
2. Only px.er_1 is eligible, and its finite values must equal its TRAIN mean,
   with std_raw and variance zero. Nonconstant or infinite data fail.
3. Parent remains untouched; survivor column order and constants stay exact.
4. All group indices and group/list/scaler file digests are derived consistently.
5. Original label-support hashes are not misrepresented as new support hashes.
6. The independent E3-M and SAC consumers reproduce the same new source probe.
7. On common valid rows, surviving tensors match exactly; newly available rows
   are counted separately. No interpolation, checkpoint promotion or fit.

The old T2/T3 choice is inherited as an engineering starting point, not proved
optimal on enlarged support. New trading metrics and new training can change.
Do not declare all checkpoint outputs bitwise equal based only on algebra or a
finite probe. No expert checkpoint is imported into SAC.
