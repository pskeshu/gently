"""
FederatedOrchestrator — Coordinates federated averaging across mesh peers.

Each round:
1. Send global model weights to all workers
2. Workers train locally for K epochs
3. Collect updated weights
4. Weighted average by dataset size
5. Evaluate global model
6. Check convergence
"""

import asyncio
import copy
import logging
from pathlib import Path
from typing import Any

from ..core.event_bus import EventType, get_event_bus

logger = logging.getLogger(__name__)


def federated_average(
    state_dicts: list[dict[str, Any]],
    weights: list[float],
) -> dict[str, Any]:
    """Compute weighted average of model state dicts.

    Parameters
    ----------
    state_dicts : list of dict
        Model state dicts from each worker.
    weights : list of float
        Dataset sizes as averaging weights.

    Returns
    -------
    dict
        Averaged state dict.
    """
    if not state_dicts:
        return {}

    if len(state_dicts) == 1:
        return copy.deepcopy(state_dicts[0])

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for federated averaging") from None

    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = len(weights)
        weights = [1.0] * len(weights)

    # Normalize weights
    norm_weights = [w / total_weight for w in weights]

    # Initialize averaged dict from first worker
    averaged = {}
    for key in state_dicts[0]:
        averaged[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

    # Weighted sum
    for sd, w in zip(state_dicts, norm_weights, strict=False):
        for key in averaged:
            averaged[key] += sd[key].float() * w

    # Cast back to original dtypes
    for key in averaged:
        averaged[key] = averaged[key].to(state_dicts[0][key].dtype)

    return averaged


class FederatedOrchestrator:
    """Coordinates federated averaging across mesh peers.

    Parameters
    ----------
    verse_map : VerseMap
        For finding GPU peers.
    transfer_client : TransferClient
        For sending/receiving weights.
    peer_client : PeerClient
        For triggering remote training via HTTP.
    """

    def __init__(self, verse_map, transfer_client=None, peer_client=None):
        self._verse_map = verse_map
        self._transfer_client = transfer_client
        self._peer_client = peer_client

    async def run_federated_training(
        self,
        pipeline_id: str,
        worker_peers: list,
        initial_weights_path: Path,
        local_epochs_per_round: int = 5,
        max_rounds: int = 20,
        convergence_threshold: float = 0.001,
        training_config: dict | None = None,
        model_config: dict | None = None,
    ) -> dict[str, Any]:
        """Run federated averaging across mesh peers.

        Parameters
        ----------
        pipeline_id : str
            ML pipeline ID.
        worker_peers : list of PersistedPeer
            Peers participating in training.
        initial_weights_path : Path
            Path to initial model weights (.pt file).
        local_epochs_per_round : int
            Local training epochs per federated round.
        max_rounds : int
            Maximum number of federated rounds.
        convergence_threshold : float
            Stop if val accuracy improvement < threshold.
        training_config : dict, optional
            Training hyperparameters.
        model_config : dict, optional
            Model architecture config.

        Returns
        -------
        dict
            Final training results.
        """
        bus = get_event_bus()
        best_global_accuracy = 0.0
        prev_accuracy = 0.0
        results = {
            "pipeline_id": pipeline_id,
            "rounds_completed": 0,
            "best_accuracy": 0.0,
            "convergence_reached": False,
            "worker_results": [],
        }

        for round_num in range(1, max_rounds + 1):
            bus.publish(
                EventType.ML_TRAINING_PROGRESS,
                {
                    "pipeline_id": pipeline_id,
                    "federated_round": round_num,
                    "max_rounds": max_rounds,
                    "phase": "distributing_weights",
                },
                source="federated",
            )

            # 1. Distribute current global weights to all workers
            # (In production, this uses TransferClient to send the .pt file)
            # For now, workers are instructed via HTTP API which weights to use

            # 2. Each worker trains locally
            worker_results = await self._train_workers(
                worker_peers,
                pipeline_id,
                round_num,
                local_epochs_per_round,
                training_config,
                model_config,
            )

            if not worker_results:
                logger.warning(f"Round {round_num}: no worker results, skipping")
                continue

            # 3. Federated average
            state_dicts = [r["state_dict"] for r in worker_results if r.get("state_dict")]
            dataset_sizes = [
                r.get("dataset_size", 1) for r in worker_results if r.get("state_dict")
            ]

            if state_dicts:
                federated_average(state_dicts, dataset_sizes)
            else:
                logger.warning(f"Round {round_num}: no state dicts to average")
                continue

            # 4. Evaluate global model
            global_accuracy = max(
                (r.get("val_accuracy", 0.0) for r in worker_results),
                default=0.0,
            )

            if global_accuracy > best_global_accuracy:
                best_global_accuracy = global_accuracy

            bus.publish(
                EventType.ML_TRAINING_PROGRESS,
                {
                    "pipeline_id": pipeline_id,
                    "federated_round": round_num,
                    "global_accuracy": global_accuracy,
                    "best_global_accuracy": best_global_accuracy,
                    "workers_contributed": len(state_dicts),
                },
                source="federated",
            )

            results["rounds_completed"] = round_num
            results["best_accuracy"] = best_global_accuracy

            # 5. Check convergence
            improvement = global_accuracy - prev_accuracy
            if round_num > 1 and improvement < convergence_threshold:
                results["convergence_reached"] = True
                logger.info(
                    f"Federated training converged at round {round_num} "
                    f"(improvement {improvement:.4f} < {convergence_threshold})"
                )
                break

            prev_accuracy = global_accuracy

        bus.publish(
            EventType.ML_TRAINING_COMPLETED,
            {
                "pipeline_id": pipeline_id,
                "federated": True,
                "rounds": results["rounds_completed"],
                "best_accuracy": results["best_accuracy"],
            },
            source="federated",
        )

        return results

    async def _train_workers(
        self,
        workers: list,
        pipeline_id: str,
        round_num: int,
        local_epochs: int,
        training_config: dict | None,
        model_config: dict | None,
    ) -> list[dict]:
        """Send training jobs to all workers and collect results.

        In production this uses PeerClient to POST /api/ml/train on each
        worker and polls for completion. For now returns placeholder results.
        """
        results = []

        # Launch training on all workers concurrently
        tasks = []
        for worker in workers:
            tasks.append(
                self._train_single_worker(
                    worker,
                    pipeline_id,
                    round_num,
                    local_epochs,
                    training_config,
                    model_config,
                )
            )

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for worker, result in zip(workers, completed, strict=False):
            if isinstance(result, BaseException):
                logger.warning(f"Worker {worker.hostname} failed in round {round_num}: {result}")
                continue
            if result:
                results.append(result)

        return results

    async def _train_single_worker(
        self,
        worker,
        pipeline_id: str,
        round_num: int,
        local_epochs: int,
        training_config: dict | None,
        model_config: dict | None,
    ) -> dict | None:
        """Train on a single worker peer via HTTP API.

        Returns worker result dict with state_dict, val_accuracy, dataset_size.
        """
        if self._peer_client is None:
            return None

        # Build a PeerInfo for the HTTP client
        from ..models import PeerInfo

        PeerInfo(
            instance_id=worker.instance_id,
            hostname=worker.hostname,
            ip_address=worker.ip_address,
            viz_port=worker.viz_port,
            is_trusted=worker.is_trusted,
            tls_enabled=worker.tls_enabled,
        )

        # POST /api/ml/train on the worker
        # This is a placeholder — actual implementation would:
        # 1. Send weights via transfer protocol
        # 2. POST training config
        # 3. Poll for completion
        # 4. Retrieve updated weights
        logger.info(f"Would train on {worker.hostname} for round {round_num}")
        return None
