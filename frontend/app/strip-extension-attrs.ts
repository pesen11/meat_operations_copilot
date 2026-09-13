/**
 * Browser extensions (Bitdefender's `bis_skin_checked` / `bis_register` /
 * `__processed_<uuid>__`, and others like it) inject attributes into the
 * server-rendered HTML before React loads. React then reports a hydration
 * mismatch for every element they touched.
 *
 * `suppressHydrationWarning` cannot fix this: it only applies to the element
 * it is set on, so it covers <body> but not the nested <div>s the extension
 * also marks, and tagging every div in the app would be whack-a-mole.
 *
 * Instead this runs as a synchronous inline script during HTML parsing - the
 * pattern Next documents in "Preventing flash before hydration" - and removes
 * the attributes before hydration compares the DOM to the server payload.
 * Nothing the app renders is touched; only attributes matching the extension
 * prefixes are removed.
 *
 * Dev-only. React logs this mismatch in development, and extra attributes on
 * an element do not trigger a re-render, so production ships none of this.
 */
export const STRIP_EXTENSION_ATTRS = `(function(){
var BAD=/^(bis_|__processed_)/i;
function clean(el){
  var a=el.attributes; if(!a) return;
  for(var i=a.length-1;i>=0;i--){var n=a[i].name; if(BAD.test(n)) el.removeAttribute(n);}
}
function sweep(root){
  clean(root);
  if(!root.querySelectorAll) return;
  var all=root.querySelectorAll('*');
  for(var i=0;i<all.length;i++) clean(all[i]);
}
sweep(document.documentElement);
var obs=new MutationObserver(function(records){
  for(var i=0;i<records.length;i++){
    var r=records[i];
    if(r.type==='attributes'){
      if(r.attributeName&&BAD.test(r.attributeName)) r.target.removeAttribute(r.attributeName);
    } else {
      for(var j=0;j<r.addedNodes.length;j++){
        if(r.addedNodes[j].nodeType===1) sweep(r.addedNodes[j]);
      }
    }
  }
});
obs.observe(document.documentElement,{attributes:true,childList:true,subtree:true});
// Keep stripping until well past hydration, then stop paying for the observer.
window.addEventListener('load',function(){setTimeout(function(){obs.disconnect();},3000);});
})();`;
