"""
FastAPI routes for the mesh subsystem.

Registered on the viz server's FastAPI app via register_mesh_routes().
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse


def register_mesh_routes(viz_server, mesh_service) -> None:
    """
    Add mesh API routes to the viz server's FastAPI app.

    Parameters
    ----------
    viz_server : VisualizationServer
        The running viz server whose .app we attach routes to.
    mesh_service : MeshService
        The mesh service instance for querying peers and local info.
    """
    router = APIRouter()

    @router.get("/api/mesh/status")
    async def mesh_status():
        """Return this node's full info (called by other peers)."""
        return JSONResponse(mesh_service.get_local_info())

    @router.get("/api/mesh/peers")
    async def mesh_peers():
        """List all discovered peers."""
        peers = mesh_service.get_peers()
        return JSONResponse({
            "peers": [p.to_dict() for p in peers],
            "count": len(peers),
        })

    @router.get("/api/mesh/peers/{instance_id}")
    async def mesh_peer_detail(instance_id: str):
        """Get specific peer details."""
        peer = mesh_service.get_peer(instance_id)
        if peer is None:
            return JSONResponse(
                {"error": f"Peer {instance_id} not found"},
                status_code=404,
            )
        return JSONResponse(peer.to_dict())

    @router.get("/api/mesh/topology")
    async def mesh_topology():
        """Full mesh view: self + all peers."""
        local = mesh_service.get_local_info()
        peers = mesh_service.get_all_peers()
        return JSONResponse({
            "self": local,
            "peers": [p.to_dict() for p in peers],
            "total_nodes": 1 + len(peers),
        })

    viz_server.app.include_router(router)
