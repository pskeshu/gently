"""
DataAssessmentEngine — Cross-network data inventory.

Local data: direct GentlyStore SQL queries (fast, full detail).
Remote data: PeerClient → data catalog API (metadata only).
Aggregates into NetworkDataInventory.
"""

import logging

from .models import NetworkDataInventory, SessionSummary

logger = logging.getLogger(__name__)


class DataAssessmentEngine:
    """Builds a unified inventory of datasets across the mesh.

    Parameters
    ----------
    gently_store : optional
        Local GentlyStore for direct SQL queries.
    peer_client : optional
        PeerClient for querying remote peers.
    verse_map : optional
        VerseMap for finding online peers with data.
    """

    def __init__(self, gently_store=None, peer_client=None, verse_map=None):
        self._store = gently_store
        self._peer_client = peer_client
        self._verse_map = verse_map

    def inventory_local(self) -> list:
        """Inventory datasets from the local GentlyStore."""
        if self._store is None:
            return []

        sessions = []
        try:
            for sess in self._store.list_sessions():
                sid = sess.session_id if hasattr(sess, "session_id") else sess.get("session_id", "")
                sname = sess.name if hasattr(sess, "name") else sess.get("name", "")

                embryos = self._store.list_embryos(sid)
                embryo_count = len(embryos)
                vol_count = 0
                annotated = 0
                gt_count = 0
                stages = set()

                for emb in embryos:
                    eid = emb.embryo_id if hasattr(emb, "embryo_id") else emb.get("embryo_id", "")
                    vols = self._store.list_volumes(sid, eid)
                    vol_count += len(vols)
                    try:
                        gts = self._store.get_ground_truth(sid, eid)
                        if gts:
                            annotated += 1
                            gt_count += len(gts)
                            for gt in gts:
                                stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                                if stage:
                                    stages.add(stage)
                    except Exception:
                        pass

                sessions.append(
                    SessionSummary(
                        session_id=sid,
                        session_name=sname,
                        embryo_count=embryo_count,
                        volume_count=vol_count,
                        annotated_embryos=annotated,
                        ground_truth_count=gt_count,
                        stages_covered=sorted(stages),
                        is_remote=False,
                    )
                )
        except Exception as e:
            logger.error(f"Local inventory failed: {e}")

        return sessions

    async def inventory_remote(self) -> tuple:
        """Inventory datasets from remote peers via data catalog API.

        Returns (sessions_list, peers_queried, peers_failed).
        """
        if self._peer_client is None or self._verse_map is None:
            return [], 0, 0

        sessions = []
        peers_queried = 0
        peers_failed = 0

        for peer_entry in self._verse_map.get_online_peers():
            peers_queried += 1
            try:
                # Build a PeerInfo-like object for PeerClient
                from ..mesh.models import PeerInfo

                peer = PeerInfo(
                    instance_id=peer_entry.instance_id,
                    hostname=peer_entry.hostname,
                    ip_address=peer_entry.ip_address,
                    viz_port=peer_entry.viz_port,
                    is_trusted=peer_entry.is_trusted,
                    tls_enabled=peer_entry.tls_enabled,
                )
                peer_sessions = await self._peer_client.fetch_peer_sessions(peer)
                if peer_sessions is None:
                    peers_failed += 1
                    continue
                for s in peer_sessions:
                    sessions.append(
                        SessionSummary(
                            session_id=s.get("session_id", ""),
                            session_name=s.get("name", ""),
                            source_peer=peer_entry.instance_id,
                            embryo_count=s.get("embryo_count", 0),
                            volume_count=s.get("volume_count", 0),
                            is_remote=True,
                        )
                    )
            except Exception as e:
                logger.debug(f"Remote inventory failed for {peer_entry.hostname}: {e}")
                peers_failed += 1

        return sessions, peers_queried, peers_failed

    async def build_inventory(self, include_remote: bool = True) -> NetworkDataInventory:
        """Build a complete network-wide data inventory."""
        local = self.inventory_local()

        remote = []
        peers_queried = 0
        peers_failed = 0
        if include_remote:
            remote, peers_queried, peers_failed = await self.inventory_remote()

        all_sessions = local + remote
        inventory = NetworkDataInventory(
            local_sessions=local,
            remote_sessions=remote,
            total_embryos=sum(s.embryo_count for s in all_sessions),
            total_volumes=sum(s.volume_count for s in all_sessions),
            total_annotated=sum(s.annotated_embryos for s in all_sessions),
            total_ground_truth=sum(s.ground_truth_count for s in all_sessions),
            peers_queried=peers_queried,
            peers_failed=peers_failed,
        )
        return inventory
