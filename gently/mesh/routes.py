"""
FastAPI routes for the mesh subsystem.

Registered on the viz server's FastAPI app via register_mesh_routes().
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from gently.core.event_bus import EventType
from gently.mesh.audit import AuditEvent


def register_mesh_routes(viz_server, mesh_service, audit_log=None) -> None:
    """
    Add mesh API routes to the viz server's FastAPI app.

    Parameters
    ----------
    viz_server : VisualizationServer
        The running viz server whose .app we attach routes to.
    mesh_service : MeshService
        The mesh service instance for querying peers and local info.
    audit_log : MeshAuditLog, optional
        Security audit logger.
    """
    router = APIRouter()

    pairing_mgr = getattr(mesh_service, "pairing_manager", None)

    # ------------------------------------------------------------------
    # Auth dependency factory — scope-aware (Phase 4)
    # ------------------------------------------------------------------

    def _make_auth_dep(required_scope: str = ""):
        """Create a FastAPI dependency that checks auth + scope."""

        async def require_mesh_auth(request: Request):
            if pairing_mgr is None:
                return  # backward compat: no pairing manager = open access
            host = request.client.host if request.client else ""
            if host in ("127.0.0.1", "::1", "localhost"):
                return  # localhost always exempt
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                peer_id = pairing_mgr.verify_token(auth[7:])
                if peer_id:
                    # Check scope if required
                    if required_scope:
                        scopes = pairing_mgr.get_scopes_for_peer(peer_id)
                        if required_scope not in scopes:
                            if audit_log:
                                audit_log.log(
                                    AuditEvent.SCOPE_DENIED,
                                    outcome="deny",
                                    peer_id=peer_id,
                                    ip=host,
                                    detail=f"scope={required_scope} path={request.url.path}",
                                )
                            if viz_server.event_bus is not None:
                                viz_server.event_bus.publish(
                                    EventType.MESH_SCOPE_DENIED,
                                    {
                                        "peer_id": peer_id,
                                        "scope": required_scope,
                                        "ip": host,
                                        "path": str(request.url.path),
                                    },
                                    source="mesh",
                                )
                            raise HTTPException(
                                status_code=403,
                                detail=f"Missing scope: {required_scope}",
                            )
                    if audit_log:
                        audit_log.log(
                            AuditEvent.AUTH_SUCCESS,
                            outcome="allow",
                            peer_id=peer_id,
                            ip=host,
                        )
                    return

            # Auth failed
            if audit_log:
                audit_log.log(
                    AuditEvent.AUTH_FAILURE,
                    outcome="deny",
                    ip=host,
                    detail=f"path={request.url.path}",
                )
            if viz_server.event_bus is not None:
                viz_server.event_bus.publish(
                    EventType.MESH_AUTH_FAILURE,
                    {"ip": host, "path": str(request.url.path)},
                    source="mesh",
                )
            raise HTTPException(status_code=403, detail="Mesh authentication required")

        return require_mesh_auth

    # ------------------------------------------------------------------
    # Authenticated mesh routes (scope: status)
    # ------------------------------------------------------------------

    @router.get("/api/mesh/status", dependencies=[Depends(_make_auth_dep("status"))])
    async def mesh_status():
        """Return this node's full info (called by other peers)."""
        info = mesh_service.get_local_info()

        # Append shared campaigns if context store is available
        cs = getattr(viz_server, "context_store", None)
        if cs is not None:
            try:
                shared = cs.get_shared_campaigns()
                shared_list = []
                for c in shared:
                    status = cs.get_plan_status(c.id)
                    shared_list.append(
                        {
                            "id": c.id,
                            "shorthand": c.shorthand,
                            "description": c.description,
                            "item_count": status["total"],
                            "completed_count": status["completed"],
                        }
                    )
                info["shared_campaigns"] = shared_list
            except Exception:
                pass

        return JSONResponse(info)

    @router.get("/api/mesh/peers", dependencies=[Depends(_make_auth_dep("status"))])
    async def mesh_peers():
        """List all discovered peers."""
        peers = mesh_service.get_peers()
        return JSONResponse(
            {
                "peers": [p.to_dict() for p in peers],
                "count": len(peers),
            }
        )

    @router.get(
        "/api/mesh/peers/{instance_id}",
        dependencies=[Depends(_make_auth_dep("status"))],
    )
    async def mesh_peer_detail(instance_id: str):
        """Get specific peer details."""
        peer = mesh_service.get_peer(instance_id)
        if peer is None:
            return JSONResponse(
                {"error": f"Peer {instance_id} not found"},
                status_code=404,
            )
        return JSONResponse(peer.to_dict())

    @router.get("/api/mesh/topology", dependencies=[Depends(_make_auth_dep("status"))])
    async def mesh_topology():
        """Full mesh view: self + all peers."""
        local = mesh_service.get_local_info()
        peers = mesh_service.get_all_peers()
        return JSONResponse(
            {
                "self": local,
                "peers": [p.to_dict() for p in peers],
                "total_nodes": 1 + len(peers),
            }
        )

    # ------------------------------------------------------------------
    # Pairing endpoints (no auth — these bootstrap trust)
    # ------------------------------------------------------------------

    @router.post("/api/mesh/pair")
    async def pair_request(request: Request):
        """Receive a pairing request from a remote peer."""
        if pairing_mgr is None:
            raise HTTPException(status_code=503, detail="Pairing not available")

        # Rate limiting
        client_ip = request.client.host if request.client else ""
        allowed, retry_after = pairing_mgr.check_rate_limit(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Too many pairing attempts. Retry after {retry_after:.0f}s",
            )
        pairing_mgr.record_attempt(client_ip)

        body = await request.json()
        initiator_id = body.get("initiator_id", "")
        hostname = body.get("hostname", "")
        nonce = body.get("nonce", "")
        initiator_cert_fp = body.get("cert_fingerprint", "")
        initiator_udp_key = body.get("udp_sign_key", "")

        if not initiator_id or not nonce:
            raise HTTPException(status_code=400, detail="initiator_id and nonce required")

        session = pairing_mgr.handle_pair_request(
            initiator_id,
            hostname,
            nonce,
            initiator_cert_fingerprint=initiator_cert_fp,
            initiator_udp_sign_key=initiator_udp_key,
        )

        if audit_log:
            audit_log.log(
                AuditEvent.PAIR_REQUESTED,
                outcome="info",
                peer_id=initiator_id,
                ip=client_ip,
                detail=f"hostname={hostname}",
            )

        # Emit event so the TUI can show the pairing notification
        if viz_server.event_bus is not None:
            viz_server.event_bus.publish(
                EventType.MESH_PAIRING_REQUESTED,
                {
                    "pairing_id": session.pairing_id,
                    "initiator_hostname": hostname,
                    "pin": session.pin,
                },
                source="mesh",
            )

        return JSONResponse(
            {
                "nonce": session.nonce_responder,
                "pairing_id": session.pairing_id,
                "status": session.status,
                "responder_id": mesh_service.instance_id,
                "responder_hostname": mesh_service._hostname,
                "cert_fingerprint": pairing_mgr.cert_fingerprint,
                "udp_sign_key": pairing_mgr.udp_sign_key,
            }
        )

    @router.get("/api/mesh/pair/{pairing_id}/status")
    async def pair_status(pairing_id: str):
        """Poll pairing session status (called by initiator)."""
        if pairing_mgr is None:
            raise HTTPException(status_code=503, detail="Pairing not available")

        session = pairing_mgr.get_session(pairing_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Pairing session not found")

        return JSONResponse(
            {
                "pairing_id": session.pairing_id,
                "status": session.status,
                "confirmed_by_initiator": session.confirmed_by_initiator,
                "confirmed_by_responder": session.confirmed_by_responder,
            }
        )

    @router.post("/api/mesh/pair/{pairing_id}/confirm")
    async def pair_confirm(pairing_id: str, request: Request):
        """Confirm pairing from one side."""
        if pairing_mgr is None:
            raise HTTPException(status_code=503, detail="Pairing not available")

        body = await request.json()
        confirmer_id = body.get("confirmer_id", "")
        if not confirmer_id:
            raise HTTPException(status_code=400, detail="confirmer_id required")

        session = pairing_mgr.confirm_pairing(pairing_id, confirmer_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Pairing session not found")

        # If both sides confirmed, mark the peer as trusted in the mesh
        if session.status == "confirmed":
            mesh_service.mark_peer_trusted(session.initiator_id)
            mesh_service.mark_peer_trusted(session.responder_id)

            if viz_server.event_bus is not None:
                # Determine the remote peer hostname
                if confirmer_id == session.initiator_id:
                    peer_hostname = session.initiator_hostname
                else:
                    peer_hostname = session.responder_hostname
                viz_server.event_bus.publish(
                    EventType.MESH_PAIRING_COMPLETED,
                    {
                        "pairing_id": session.pairing_id,
                        "peer_hostname": peer_hostname,
                    },
                    source="mesh",
                )

        return JSONResponse(
            {
                "pairing_id": session.pairing_id,
                "status": session.status,
            }
        )

    @router.post("/api/mesh/pair/{pairing_id}/reject")
    async def pair_reject(pairing_id: str):
        """Reject a pairing session."""
        if pairing_mgr is None:
            raise HTTPException(status_code=503, detail="Pairing not available")

        session = pairing_mgr.reject_pairing(pairing_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Pairing session not found")

        return JSONResponse(
            {
                "pairing_id": session.pairing_id,
                "status": session.status,
            }
        )

    # ------------------------------------------------------------------
    # Verse map routes (scope: status)
    # ------------------------------------------------------------------

    @router.get("/api/mesh/verse-map", dependencies=[Depends(_make_auth_dep("status"))])
    async def verse_map():
        """Full persistent topology — includes offline peers."""
        vm = mesh_service.verse_map
        peers = vm.get_all_peers()
        return JSONResponse(
            {
                "peers": [p.to_dict() for p in peers],
                "online_count": len(vm.get_online_peers()),
                "offline_count": len(vm.get_offline_peers()),
                "total_count": len(peers),
            }
        )

    @router.get(
        "/api/mesh/verse-map/resources/{capability}",
        dependencies=[Depends(_make_auth_dep("status"))],
    )
    async def verse_map_resources(capability: str):
        """Find peers matching a capability (route-finding)."""
        vm = mesh_service.verse_map
        peers = vm.find_resource(capability)
        return JSONResponse(
            {
                "capability": capability,
                "peers": [p.to_dict() for p in peers],
                "count": len(peers),
            }
        )

    # ------------------------------------------------------------------
    # Data catalog routes (scope: data)
    # ------------------------------------------------------------------

    @router.get("/api/data/sessions", dependencies=[Depends(_make_auth_dep("data"))])
    async def data_sessions():
        """List all sessions with counts."""
        store = getattr(viz_server, "gently_store", None)
        if store is None:
            return JSONResponse({"sessions": [], "count": 0})
        try:
            sessions = store.list_sessions()
            result = []
            for s in sessions:
                sid = s.session_id if hasattr(s, "session_id") else s.get("session_id", "")
                name = s.name if hasattr(s, "name") else s.get("name", "")
                created = s.created_at if hasattr(s, "created_at") else s.get("created_at", "")
                last_active = (
                    s.last_active if hasattr(s, "last_active") else s.get("last_active", "")
                )
                embryos = store.list_embryos(sid)
                vol_count = 0
                for e in embryos:
                    eid = e.embryo_id if hasattr(e, "embryo_id") else e.get("embryo_id", "")
                    vol_count += len(store.list_volumes(sid, eid))
                result.append(
                    {
                        "session_id": sid,
                        "name": name,
                        "embryo_count": len(embryos),
                        "volume_count": vol_count,
                        "created_at": created,
                        "last_active": last_active,
                    }
                )
            return JSONResponse({"sessions": result, "count": len(result)})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get(
        "/api/data/sessions/{session_id}",
        dependencies=[Depends(_make_auth_dep("data"))],
    )
    async def data_session_detail(session_id: str):
        """Detailed session info with embryo list."""
        store = getattr(viz_server, "gently_store", None)
        if store is None:
            raise HTTPException(status_code=404, detail="No data store")
        try:
            session = store.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            embryos = store.list_embryos(session_id)
            embryo_list = []
            total_vols = 0
            for e in embryos:
                eid = e.embryo_id if hasattr(e, "embryo_id") else e.get("embryo_id", "")
                nickname = e.nickname if hasattr(e, "nickname") else e.get("nickname", "")
                vols = store.list_volumes(session_id, eid)
                vol_count = len(vols)
                total_vols += vol_count
                has_gt = False
                stages = []
                try:
                    gts = store.get_ground_truth(session_id, eid)
                    has_gt = len(gts) > 0
                    stages = list(
                        {(gt.stage if hasattr(gt, "stage") else gt.get("stage", "")) for gt in gts}
                    )
                except Exception:
                    pass
                embryo_list.append(
                    {
                        "embryo_id": eid,
                        "nickname": nickname,
                        "volume_count": vol_count,
                        "has_ground_truth": has_gt,
                        "stages_annotated": stages,
                    }
                )
            sname = session.name if hasattr(session, "name") else session.get("name", "")
            return JSONResponse(
                {
                    "session_id": session_id,
                    "name": sname,
                    "embryos": embryo_list,
                    "total_volumes": total_vols,
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/api/data/coverage", dependencies=[Depends(_make_auth_dep("data"))])
    async def data_coverage():
        """Annotation coverage summary across all sessions."""
        store = getattr(viz_server, "gently_store", None)
        if store is None:
            return JSONResponse(
                {
                    "total_embryos": 0,
                    "annotated_embryos": 0,
                    "coverage_pct": 0.0,
                    "stage_counts": {},
                    "imbalance_ratio": 0.0,
                    "gaps": [],
                }
            )
        try:
            sessions = store.list_sessions()
            total_embryos = 0
            annotated_embryos = 0
            stage_counts = {}
            for s in sessions:
                sid = s.session_id if hasattr(s, "session_id") else s.get("session_id", "")
                embryos = store.list_embryos(sid)
                total_embryos += len(embryos)
                for e in embryos:
                    eid = e.embryo_id if hasattr(e, "embryo_id") else e.get("embryo_id", "")
                    try:
                        gts = store.get_ground_truth(sid, eid)
                        if gts:
                            annotated_embryos += 1
                            for gt in gts:
                                stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                                if stage:
                                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    except Exception:
                        pass
            coverage_pct = (annotated_embryos / total_embryos * 100) if total_embryos else 0.0
            counts = list(stage_counts.values())
            imbalance_ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else 0.0
            # Find stages with notably low counts
            avg = sum(counts) / len(counts) if counts else 0
            gaps = [s for s, c in stage_counts.items() if c < avg * 0.5]
            return JSONResponse(
                {
                    "total_embryos": total_embryos,
                    "annotated_embryos": annotated_embryos,
                    "coverage_pct": round(coverage_pct, 1),
                    "stage_counts": stage_counts,
                    "imbalance_ratio": round(imbalance_ratio, 2),
                    "gaps": gaps,
                }
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @router.get("/api/data/stages", dependencies=[Depends(_make_auth_dep("data"))])
    async def data_stages():
        """Stage distribution across all sessions."""
        store = getattr(viz_server, "gently_store", None)
        if store is None:
            return JSONResponse({"stage_distribution": {}, "by_session": {}})
        try:
            sessions = store.list_sessions()
            total_dist = {}
            by_session = {}
            for s in sessions:
                sid = s.session_id if hasattr(s, "session_id") else s.get("session_id", "")
                session_dist = {}
                embryos = store.list_embryos(sid)
                for e in embryos:
                    eid = e.embryo_id if hasattr(e, "embryo_id") else e.get("embryo_id", "")
                    try:
                        gts = store.get_ground_truth(sid, eid)
                        for gt in gts:
                            stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                            if stage:
                                total_dist[stage] = total_dist.get(stage, 0) + 1
                                session_dist[stage] = session_dist.get(stage, 0) + 1
                    except Exception:
                        pass
                if session_dist:
                    by_session[sid] = session_dist
            return JSONResponse(
                {
                    "stage_distribution": total_dist,
                    "by_session": by_session,
                }
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    viz_server.app.include_router(router)
