const $ = id => document.getElementById(id);
const world = $('world'), ctx = world.getContext('2d');
const chart = $('chart'), plot = chart.getContext('2d');
let state = null, playing = false, repeating = false, busy = false, resetting = false, lastStep = 0;
let azimuth = -0.62, elevation = 0.83, zoom = 1, dragging = null;
let center = [0, 0, 1.4], trails = [], history = [], lastTime = -1;
let queue = Promise.resolve();
const colors = {drone:'#83ddc7', target:'#e6be7c', obstacle:'#dc918b', grid:'#20323e', muted:'#65818e'};

function notify(message) { $('error').textContent = message; $('error').hidden = !message; }
function api(path, payload) {
  const run = async () => {
    const response = await fetch(path, payload === undefined ? {} : {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
    return value;
  };
  const next = queue.then(run);
  queue = next.catch(() => {});
  return next;
}

function liveError(s) {
  return Math.sqrt(s.positions.reduce((sum,p,i) => sum+p.reduce((e,x,k)=>e+(x-s.slots[i][k])**2,0),0)/s.positions.length);
}

function accept(s) {
  state = s;
  if (s.time < lastTime) { trails=[]; history=[]; lastTime=-1; }
  if (s.time > lastTime) {
    s.positions.forEach((p,i) => { (trails[i] ||= []).push([...p]); if (trails[i].length>110) trails[i].shift(); });
    history.push([s.time,liveError(s)]);
    history = history.filter(p=>p[0]>=s.time-12);
    lastTime=s.time;
  }
  $('connection').textContent='Local simulation connected';
  $('policy').textContent=s.policy;
  $('learning-note').hidden=s.policy!=='learned checkpoint';
  $('model-note').textContent=`Reduced-order dynamics · illustrative drone attitude · ${s.policy}`;
  $('simulation-time').textContent=`T + ${s.time.toFixed(2).padStart(5,'0')} s`;
  $('slot-error').textContent=liveError(s).toFixed(3);
  $('clearance').textContent=s.metrics.min_surface_clearance_m===null ? '—' : s.metrics.min_surface_clearance_m.toFixed(3);
  const events=s.metrics.events, recovered=events.filter(e=>e.recovery_s!==null).length;
  $('recovered').textContent=`${recovered} / ${events.length}`;
  $('state-note').textContent=`Seed ${s.seed} · ${s.positions.length} agents`;
  $('mission-title').textContent=$('scenario').querySelector(`option[value="${s.scenario}"]`).textContent;
  const status=s.metrics.failure ? 'Failed' : !s.running ? 'Complete' : !playing ? 'Paused' : s.obstacles.length ? 'Avoiding' : liveError(s)>.2 ? 'Recovering' : 'Following';
  $('flight-state').textContent=status;
  $('flight-state').style.color=s.metrics.failure ? colors.obstacle : colors.drone;
  $('phase').textContent=s.metrics.failure ? `FLIGHT STOPPED / ${s.metrics.failure.toUpperCase()}` : !s.running ? 'EPISODE COMPLETE' : playing ? 'SIMULATION RUNNING' : 'READY / PAUSED';
  if (!s.running) playing=false;
  $('play').textContent=resetting?'Preparing…':playing?'Pause flight':s.running?'Start flight':'Run again';
  $('play').disabled=resetting;
  $('reset').disabled=resetting;
  $('repeat').disabled=resetting;
  $('repeat').textContent=`Repeat flights: ${repeating?'On':'Off'}`;
  $('repeat').setAttribute('aria-pressed',String(repeating));
  $('inject').disabled=$('headon').disabled=resetting||!s.running;
}

async function reset(start=false) {
  if(resetting)return;
  resetting=true;
  playing=false;
  if(!start)repeating=false;
  if(state)accept(state);
  try {
    notify('');
    const s=await api('/api/reset',{scenario:$('scenario').value,drones:Number($('drones').value),seed:Number($('seed').value)});
    trails=[];history=[];lastTime=-1;
    center=[s.target[0]-2,0,1.4];
    playing=start;
    accept(s);
  } catch(e) {repeating=false;notify(e.message);}
  finally {resetting=false;if(state)accept(state);}
}
$('reset').onclick=()=>reset();
$('play').onclick=()=>{
  if(!state||resetting)return;
  if(!state.running){reset(true);return;}
  playing=!playing;
  if(!playing)repeating=false;
  accept(state);
};
$('repeat').onclick=()=>{
  if(!state||resetting)return;
  repeating=!repeating;
  if(repeating&&!state.running){reset(true);return;}
  if(repeating)playing=true;
  accept(state);
};
async function inject(angle) {
  try {notify('');accept(await api('/api/inject',{speed:Number($('speed').value),radius:Number($('radius').value),angle}));}
  catch(e){notify(e.message);}
}
$('inject').onclick=()=>inject(Math.PI/2);
$('headon').onclick=()=>inject(Math.PI);
for (const [id,unit] of [['speed','m/s'],['radius','m']]) $(id).oninput=()=>{$(`${id}-value`).textContent=`${Number($(id).value).toFixed(1)} ${unit}`;};
$('top').onclick=()=>{elevation=Math.PI/2;azimuth=0;$('top').classList.add('selected');$('orbit').classList.remove('selected');};
$('orbit').onclick=()=>{elevation=.83;azimuth=-.62;$('orbit').classList.add('selected');$('top').classList.remove('selected');};
$('export').onclick=()=>{
  if(!state)return;
  const output=document.createElement('canvas');output.width=world.width;output.height=world.height;
  const brush=output.getContext('2d');brush.fillStyle='#0e161c';brush.fillRect(0,0,output.width,output.height);brush.drawImage(world,0,0);
  const ratio=window.devicePixelRatio||1;brush.scale(ratio,ratio);brush.fillStyle='#e5edee';brush.font='12px monospace';
  brush.fillText(`SwarmFlow / ${state.scenario} / ${state.policy} / seed ${state.seed} / ${state.time.toFixed(2)} s`,18,24);
  brush.fillStyle='#889ba4';brush.font='10px monospace';brush.fillText('3D stabilized-flight model · illustrative attitude',18,42);
  output.toBlob(blob=>{const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=`swarmflow-${state.scenario}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});
};
world.onpointerdown=e=>{dragging=[e.clientX,e.clientY];world.setPointerCapture(e.pointerId);};
world.onpointermove=e=>{if(dragging){azimuth+=(e.clientX-dragging[0])*.006;elevation=Math.max(.15,Math.min(1.57,elevation+(e.clientY-dragging[1])*.006));dragging=[e.clientX,e.clientY];}};
world.onpointerup=()=>dragging=null;
world.onwheel=e=>{e.preventDefault();zoom=Math.max(.45,Math.min(2,zoom*Math.exp(-e.deltaY*.001)));};

function fit(canvas, context) {
  const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  context.setTransform(ratio,0,0,ratio,0,0);
  return [w,h];
}
let w=0,h=0,scale=1;
function project(p){
  const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
  const u=x*Math.cos(azimuth)-y*Math.sin(azimuth),v=x*Math.sin(azimuth)+y*Math.cos(azimuth);
  return [w*.5+u*scale,h*.53+(v*Math.sin(elevation)-z*Math.cos(elevation))*scale,v*Math.cos(elevation)+z*Math.sin(elevation)];
}
function line(a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);}
function circle(p,r,fill,stroke){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function drawDrone(p,index,t){
  const a=state.accelerations[index], tiltX=-a[0]*.025, tiltY=-a[1]*.025;
  const rotorPoints=[[-.23,-.23],[.23,.23],[-.23,.23],[.23,-.23]].map(([x,y])=>[p[0]+x,p[1]+y,p[2]+tiltX*x+tiltY*y]);
  line(rotorPoints[0],rotorPoints[1],colors.drone,2);line(rotorPoints[2],rotorPoints[3],colors.drone,2);
  for(const r of rotorPoints){
    const point=project(r);circle(point,Math.max(2.5,scale*.12),'#18332f',colors.drone);
    const theta=t*.045+index,rr=scale*.09;
    ctx.beginPath();ctx.moveTo(point[0]+Math.cos(theta)*rr,point[1]+Math.sin(theta)*rr*.4);ctx.lineTo(point[0]-Math.cos(theta)*rr,point[1]-Math.sin(theta)*rr*.4);ctx.strokeStyle='#83ddc777';ctx.stroke();
  }
  const point=project(p);circle(point,Math.max(2,scale*.075),colors.drone);
  ctx.fillStyle='#bedbd8';ctx.font='9px ui-monospace,monospace';ctx.fillText(String(index+1).padStart(2,'0'),point[0]+10,point[1]-9);
}
function drawObject(obj){
  const p=project(obj.position),r=obj.radius*scale;
  circle(p,r,'#dc918b27',colors.obstacle);
  ctx.strokeStyle='#dc918b77';ctx.beginPath();ctx.ellipse(p[0],p[1],r,r*.3,0,0,Math.PI*2);ctx.stroke();
  ctx.beginPath();ctx.ellipse(p[0],p[1],r*.35,r,0,0,Math.PI*2);ctx.stroke();
  const end=obj.position.map((v,k)=>v+obj.velocity[k]*.7);line(obj.position,end,'#dc918b88',1,[3,3]);
}
function draw(t){
  [w,h]=fit(world,ctx);ctx.clearRect(0,0,w,h);if(!state)return;
  scale=Math.min(w/17,h/10)*zoom;
  center[0]+=(state.target[0]-2-center[0])*.04;
  center[1]+=(state.target[1]-center[1])*.04;
  const baseX=Math.floor(center[0]),baseY=Math.floor(center[1]);
  for(let i=-14;i<=14;i++){
    line([baseX+i,baseY-14,0],[baseX+i,baseY+14,0],i%5===0?'#2a424f':colors.grid,.7);
    line([baseX-14,baseY+i,0],[baseX+14,baseY+i,0],i%5===0?'#2a424f':colors.grid,.7);
  }
  const origin=project([baseX-5,baseY-4,0]);ctx.fillStyle=colors.muted;ctx.font='8px ui-monospace,monospace';ctx.fillText('1 m GRID',origin[0],origin[1]);
  trails.forEach(points=>{for(let k=1;k<points.length;k++)line(points[k-1],points[k],'#80d4c226',1);});
  state.slots.forEach((p,i)=>{
    const q=project(p);ctx.strokeStyle='#819da378';ctx.lineWidth=1;ctx.strokeRect(q[0]-4,q[1]-4,8,8);
    line(p,state.positions[i],'#90beb347',.8,[3,4]);
    const shadow=project([state.positions[i][0],state.positions[i][1],0]);circle(shadow,scale*.17,'#70b9b211');
  });
  const tp=project(state.target);ctx.save();ctx.translate(tp[0],tp[1]);ctx.rotate(Math.PI/4);ctx.fillStyle='#e6be7c20';ctx.strokeStyle=colors.target;ctx.lineWidth=1.4;ctx.fillRect(-6,-6,12,12);ctx.strokeRect(-6,-6,12,12);ctx.restore();
  ctx.fillStyle=colors.target;ctx.font='9px ui-monospace,monospace';ctx.fillText('REFERENCE',tp[0]+13,tp[1]+3);
  line(state.target,[state.target[0]+.8,state.target[1],state.target[2]],'#e6be7c66',1,[2,3]);
  const objects=state.positions.map((p,i)=>({depth:project(p)[2],draw:()=>drawDrone(p,i,t)}));
  state.obstacles.forEach(o=>objects.push({depth:project(o.position)[2],draw:()=>drawObject(o)}));
  objects.sort((a,b)=>a.depth-b.depth).forEach(o=>o.draw());
  drawChart();
}
function drawChart(){
  const [cw,ch]=fit(chart,plot);plot.clearRect(0,0,cw,ch);if(!history.length)return;
  const max=Math.max(.5,...history.map(p=>p[1]))*1.1,left=30,right=cw-8,bottom=ch-16,top=7;
  plot.font='8px ui-monospace,monospace';plot.fillStyle=colors.muted;plot.strokeStyle=colors.grid;plot.lineWidth=1;
  for(const y of [0,max/2,max]){const yy=bottom-y/max*(bottom-top);plot.beginPath();plot.moveTo(left,yy);plot.lineTo(right,yy);plot.stroke();plot.fillText(y.toFixed(1),0,yy+3);}
  const start=Math.max(0,state.time-12),end=Math.max(12,state.time);
  plot.fillText(`${start.toFixed(0)} s`,left,ch-1);plot.fillText(`${end.toFixed(0)} s`,right-20,ch-1);
  plot.beginPath();history.forEach((p,i)=>{const x=left+(p[0]-start)/(end-start)*(right-left),y=bottom-p[1]/max*(bottom-top);i?plot.lineTo(x,y):plot.moveTo(x,y);});plot.strokeStyle=colors.drone;plot.lineWidth=1.5;plot.stroke();
}
async function tick(t){
  draw(t);
  if(playing&&!busy&&t-lastStep>=95){
    busy=true;lastStep=t;
    try{
      const next=await api('/api/step',{steps:2});
      accept(next);
      if(!next.running&&repeating)await reset(true);
    }
    catch(e){playing=false;repeating=false;if(state)accept(state);notify(e.message);$('connection').textContent='Connection interrupted';}
    finally{busy=false;}
  }
  requestAnimationFrame(tick);
}
api('/api/state').then(s=>{
  $('scenario').value=s.scenario;$('drones').value=String(s.positions.length);$('seed').value=s.seed;
  center=[s.target[0]-2,0,1.4];accept(s);
}).catch(e=>notify(`Could not connect to the simulator: ${e.message}. Keep the Python server running and reload.`));
requestAnimationFrame(tick);
