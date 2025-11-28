from symop_proto.core.protocols import (
    EnvelopeLike,
    SupportsOverlapWIthGeneric,
)


class OpaqueEnv(EnvelopeLike):
    def overlap(self, other: EnvelopeLike) -> complex:
        # Never called; BaseEnvelope should raise before delegating here
        raise AssertionError("Should not be called")


class HookEnv(SupportsOverlapWIthGeneric):
    def overlap(self, other: EnvelopeLike) -> complex:
        # Should not be chosen (BaseEnvelope prefers overlap_with_generic when *other* is hooky)
        return 999.0 + 0j

    def overlap_with_generic(self, other: EnvelopeLike) -> complex:
        # This is the one BaseEnvelope should call
        return 0.123 + 0.0j
