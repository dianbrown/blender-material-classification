import bpy, json, sys
from pathlib import Path

def nodes(tree):
    if tree is None: return []
    return [dict(name=n.name,type=n.bl_idname,image=n.image.filepath if getattr(n,'image',None) else None,group=getattr(getattr(n,'node_tree',None),'name',None)) for n in tree.nodes]
r = dict(blender_version=bpy.app.version_string,
 objects=[dict(name=o.name,type=o.type,library=o.library.filepath if o.library else None,hide_render=o.hide_render,materials=[s.material.name if s.material else None for s in o.material_slots],vertices=len(o.data.vertices) if o.type=='MESH' else None,polygons=len(o.data.polygons) if o.type=='MESH' else None,modifiers=[dict(name=m.name,type=m.type) for m in o.modifiers]) for o in bpy.data.objects],
 images=[dict(name=i.name,source=i.source,path=i.filepath,packed=bool(i.packed_file),users=i.users) for i in bpy.data.images],
 materials=[dict(name=m.name,base_material_id=m.get('base_material_id'),nodes=nodes(m.node_tree)) for m in bpy.data.materials],
 worlds=[dict(name=w.name,nodes=nodes(w.node_tree)) for w in bpy.data.worlds],
 libraries=[l.filepath for l in bpy.data.libraries],
 movieclips=[c.filepath for c in bpy.data.movieclips],sounds=[s.filepath for s in bpy.data.sounds],fonts=[f.filepath for f in bpy.data.fonts],cache_files=[c.filepath for c in bpy.data.cache_files],
 node_groups=[dict(name=g.name,nodes=nodes(g)) for g in bpy.data.node_groups],
 scenes=[dict(name=s.name,world=s.world.name if s.world else None,nodes=nodes(getattr(s,'node_tree',None)),engine=s.render.engine) for s in bpy.data.scenes])
Path(sys.argv[-1]).write_text(json.dumps(r,indent=2))
print('SCENE_INSPECTION_DONE',len(r['images']),len(r['materials']))
