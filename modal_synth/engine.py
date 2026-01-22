from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def normalize(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(np.abs(v)) + 1e-12)
    return v / s


def drive_distribution_attack(K: int) -> np.ndarray:
    # brighter injection profile
    k = np.arange(K, dtype=np.float32) + 1.0
    return normalize(1.0 / (k ** 0.6))


def drive_distribution_sustain(K: int) -> np.ndarray:
    # duller injection profile
    k = np.arange(K, dtype=np.float32) + 1.0
    return normalize(1.0 / (k ** 1.2))


@dataclass
class ArticulationParams:
    # ADSR-like UI, but physical mapping
    attack_ms: float = 10.0          # contact release time (reverse damping relax)
    decay_ms: float = 25.0           # excitation impulse decay
    sustain: float = 0.2             # sustain hand pressure (damping multiplier target)
    release_ms: float = 60.0         # release time constant
    release_gesture: float = 0.3     # 0=let-go, 1=mute
    gamma_attack: float = 12.0       # strong contact damping at note-on
    drive_morph_ms: float = 20.0     # attack->sustain drive morph time


@dataclass
class NodeNetworkParams:
    N: int = 5
    K: int = 4
    sample_rate: int = 48000

    # ring coupling strength
    kappa: float = 0.05

    # hand frequency profile: damps higher modes more
    hand_damp_alpha: float = 0.6
    hand_damp_power: float = 1.0


@dataclass
class Voice:
    active: bool = False
    gate: bool = False
    releasing: bool = False

    note: int = 60
    velocity: float = 1.0
    node_mask: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float32))

    t_on: float = 0.0
    t_rel: float = 0.0
    gamma_target_release: float = 1.0

    def note_on(self, note: int, velocity: float, node_mask: np.ndarray) -> None:
        self.active = True
        self.gate = True
        self.releasing = False
        self.note = int(note)
        self.velocity = float(velocity)
        self.node_mask = node_mask.astype(np.float32)
        self.t_on = 0.0
        self.t_rel = 0.0

    def note_off(self, art: ArticulationParams) -> None:
        if not self.active or not self.gate:
            return
        self.gate = False
        self.releasing = True
        self.t_rel = 0.0

        # let-go -> 1.0, mute -> gamma_attack (strong)
        mute_gamma = art.gamma_attack
        self.gamma_target_release = (1.0 - art.release_gesture) * 1.0 + art.release_gesture * mute_gamma

    def step_controls(self, dt: float, art: ArticulationParams) -> Tuple[float, float, float]:
        """
        Returns:
          E: excitation strength
          Gamma: hand damping multiplier
          alpha: drive morph 0..1 (0=attack profile, 1=sustain profile)
        """
        tau_open = max(art.attack_ms, 0.1) / 1000.0
        tau_imp = max(art.decay_ms, 0.1) / 1000.0
        tau_morph = max(art.drive_morph_ms, 0.1) / 1000.0
        tau_rel = max(art.release_ms, 0.1) / 1000.0

        if self.active and self.gate:
            self.t_on += dt

            # excitation impulse
            E0 = self.velocity
            E = E0 * float(np.exp(-self.t_on / tau_imp))

            # reverse damping: start high then relax to sustain pressure
            Gamma = art.sustain + (art.gamma_attack - art.sustain) * float(np.exp(-self.t_on / tau_open))

            # drive profile morph
            alpha = 1.0 - float(np.exp(-self.t_on / tau_morph))
            return E, Gamma, alpha

        if self.active and self.releasing:
            self.t_rel += dt
            E = 0.0

            # move Gamma toward release target
            Gamma0 = art.sustain  # approximation; good enough for v1
            Gamma = self.gamma_target_release + (Gamma0 - self.gamma_target_release) * float(np.exp(-self.t_rel / tau_rel))

            # deactivate after a few time constants
            if self.t_rel > 6.0 * tau_rel:
                self.active = False
                self.releasing = False

            return E, Gamma, 1.0

        return 0.0, 0.0, 1.0


@dataclass
class NodeNetwork:
    params: NodeNetworkParams
    a: np.ndarray = field(init=False)          # (N,K) complex
    omega: np.ndarray = field(init=False)      # (N,K) float
    gamma_body: np.ndarray = field(init=False) # (N,K) float
    weight: np.ndarray = field(init=False)     # (N,K) float

    def __post_init__(self) -> None:
        p = self.params
        self.a = np.zeros((p.N, p.K), dtype=np.complex64)

        base_mult = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)[:p.K]
        base_gamma = np.array([0.6, 0.8, 1.1, 1.6], dtype=np.float32)[:p.K]
        base_weight = np.array([1.0, 0.7, 0.5, 0.35], dtype=np.float32)[:p.K]

        self.omega = np.zeros((p.N, p.K), dtype=np.float32)
        self.gamma_body = np.zeros((p.N, p.K), dtype=np.float32)
        self.weight = np.zeros((p.N, p.K), dtype=np.float32)

        for j in range(p.N):
            wj = 1.0 + 0.02 * (j - (p.N - 1) / 2.0)
            self.gamma_body[j, :] = base_gamma * (1.0 + 0.05 * j)
            self.weight[j, :] = base_weight * wj

        self.omega[:, :] = 2.0 * np.pi * 220.0 * base_mult[None, :]

    def hand_profile_Hk(self) -> np.ndarray:
        p = self.params
        k = np.arange(p.K, dtype=np.float32)
        denom = max(p.K - 1, 1)
        return 1.0 + p.hand_damp_alpha * ((k / denom) ** p.hand_damp_power)

    def ring_coupling(self, a_snapshot: np.ndarray) -> np.ndarray:
        p = self.params
        left = np.roll(a_snapshot, 1, axis=0)
        right = np.roll(a_snapshot, -1, axis=0)
        avg = 0.5 * (left + right)
        return p.kappa * (avg - a_snapshot)

    def step_one(self, dt: float, drive: np.ndarray, gamma_hand: np.ndarray, omega: np.ndarray) -> None:
        """
        drive: (N,K) complex (real drive ok)
        gamma_hand: (N,) float (>=1 baseline)
        omega: (N,K) float rad/s
        """
        Hk = self.hand_profile_Hk()[None, :]  # (1,K)
        a0 = self.a.copy()
        coup = self.ring_coupling(a0)

        gamma_eff = self.gamma_body * (gamma_hand[:, None]) * Hk
        da = ((-gamma_eff + 1j * omega) * a0) + drive + coup

        # explicit Euler (v1)
        self.a = (a0 + dt * da).astype(np.complex64)

    def render_mono(self) -> float:
        return float(np.sum(self.weight * np.real(self.a)))


@dataclass
class EngineConfig:
    sample_rate: int = 48000
    N: int = 5
    K: int = 4
    voice_count: int = 16


class Engine:
    """
    Shared-body engine:
      - Nodes (body) persist and hold state.
      - Voices are gesture controllers (E(t), Gamma(t), routing mask).
    """
    def __init__(self, cfg: EngineConfig, art: Optional[ArticulationParams] = None, body: Optional[NodeNetworkParams] = None):
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.dt = 1.0 / self.sr

        self.art = art or ArticulationParams()
        body_params = body or NodeNetworkParams()
        body_params.sample_rate = self.sr
        body_params.N = cfg.N
        body_params.K = cfg.K

        self.body = NodeNetwork(body_params)

        self.voices: List[Voice] = [Voice(node_mask=np.ones(cfg.N, dtype=np.float32)) for _ in range(cfg.voice_count)]

        self.base_mult = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)[:cfg.K]
        self.D_att = drive_distribution_attack(cfg.K)
        self.D_sus = drive_distribution_sustain(cfg.K)

        self._last_pitch_hz = 110.0

    def alloc_voice(self) -> Voice:
        for v in self.voices:
            if not v.active:
                return v
        for v in self.voices:
            if v.releasing:
                return v
        return self.voices[0]

    def note_on(self, note: int, vel: float, node_mask: np.ndarray) -> None:
        v = self.alloc_voice()
        v.note_on(note, vel, node_mask)
        self._last_pitch_hz = midi_to_hz(note)

    def note_off(self, note: int) -> None:
        for v in self.voices:
            if v.active and v.note == int(note) and v.gate:
                v.note_off(self.art)

    def step_one_sample(self) -> float:
        N, K = self.cfg.N, self.cfg.K

        drive = np.zeros((N, K), dtype=np.complex64)
        gamma_hand = np.zeros((N,), dtype=np.float32)

        # pitch reference (v1): most recent played note
        pitch_hz = self._last_pitch_hz
        omega = 2.0 * np.pi * pitch_hz * self.base_mult[None, :]
        omega = np.repeat(omega, N, axis=0).astype(np.float32)

        # accumulate from voices
        for v in self.voices:
            if not v.active:
                continue
            E, Gamma, alpha = v.step_controls(self.dt, self.art)
            if E == 0.0 and Gamma == 0.0:
                continue

            # damping combine rule = max per node
            gamma_hand = np.maximum(gamma_hand, (Gamma * v.node_mask).astype(np.float32))

            D = (1.0 - alpha) * self.D_att + alpha * self.D_sus  # (K,)
            for j in range(N):
                m = float(v.node_mask[j])
                if m <= 0.0:
                    continue
                drive[j, :] += (E * m) * D.astype(np.complex64)

        # default baseline damping multiplier = 1.0 if no voice touches node
        gamma_hand = np.where(gamma_hand > 0.0, gamma_hand, 1.0).astype(np.float32)

        self.body.step_one(self.dt, drive, gamma_hand, omega)
        return self.body.render_mono()
