"use strict";(self.webpackChunkapp=self.webpackChunkapp||[]).push([[2584],{2584:(O,l,s)=>{s.r(l),s.d(l,{HomePageModule:()=>_});var a=s(6895),i=s(4556),g=s(433),c=s(2598),e=s(8256),m=s(2468);function p(n,o){1&n&&e._UZ(0,"ion-icon",6)}function d(n,o){if(1&n&&(e.TgZ(0,"ion-item",1),e._UZ(1,"div",2),e.TgZ(2,"ion-label",3)(3,"h2"),e._uU(4),e.TgZ(5,"span",4)(6,"ion-note"),e._uU(7),e.qZA(),e.YNc(8,p,1,0,"ion-icon",5),e.qZA()(),e.TgZ(9,"h3"),e._uU(10),e.qZA()()()),2&n){const t=e.oxw();e.Q6J("routerLink","/message/"+t.exercise.id)("detail",!1),e.xp6(1),e.Tol(t.exercise.read?"dot":"dot dot-unread"),e.xp6(3),e.hij(" ",t.exercise.name," "),e.xp6(3),e.Oqu(t.exercise.index),e.xp6(1),e.Q6J("ngIf",t.isIos()),e.xp6(2),e.Oqu(t.exercise.question)}}let u=(()=>{class n{isIos(){const t=window;return t&&t.Ionic&&"ios"===t.Ionic.mode}}return n.\u0275fac=function(t){return new(t||n)},n.\u0275cmp=e.Xpm({type:n,selectors:[["app-message"]],inputs:{exercise:"exercise"},decls:1,vars:1,consts:[[3,"routerLink","detail",4,"ngIf"],[3,"routerLink","detail"],["slot","start"],[1,"ion-text-wrap"],[1,"date"],["name","chevron-forward","size","small",4,"ngIf"],["name","chevron-forward","size","small"]],template:function(t,r){1&t&&e.YNc(0,d,11,8,"ion-item",0),2&t&&e.Q6J("ngIf",r.exercise)},dependencies:[a.O5,i.gu,i.Ie,i.Q$,i.uN,i.YI,c.rH],styles:["ion-item[_ngcontent-%COMP%]{--padding-start: 0;--inner-padding-end: 0}ion-label[_ngcontent-%COMP%]{margin-top:12px;margin-bottom:12px}ion-item[_ngcontent-%COMP%]   h2[_ngcontent-%COMP%]{font-weight:600;margin:0}ion-item[_ngcontent-%COMP%]   p[_ngcontent-%COMP%]{text-overflow:ellipsis;overflow:hidden;white-space:nowrap;width:95%}ion-item[_ngcontent-%COMP%]   .date[_ngcontent-%COMP%]{float:right;align-items:center;display:flex}ion-item[_ngcontent-%COMP%]   ion-icon[_ngcontent-%COMP%]{color:#c9c9ca}ion-item[_ngcontent-%COMP%]   ion-note[_ngcontent-%COMP%]{font-size:15px;margin-right:8px;font-weight:400}ion-item[_ngcontent-%COMP%]   ion-note.md[_ngcontent-%COMP%]{margin-right:14px}.dot[_ngcontent-%COMP%]{display:block;height:12px;width:12px;border-radius:50%;align-self:start;margin:16px 10px 16px 16px}.dot-unread[_ngcontent-%COMP%]{background:var(--ion-color-primary)}ion-footer[_ngcontent-%COMP%]   ion-title[_ngcontent-%COMP%]{font-size:11px;font-weight:400}"]}),n})();function f(n,o){1&n&&e._UZ(0,"app-message",6),2&n&&e.Q6J("exercise",o.$implicit)}function x(n,o){1&n&&e._UZ(0,"app-message",6),2&n&&e.Q6J("exercise",o.$implicit)}const h=[{path:"",component:(()=>{class n{constructor(t){this.data=t}refresh(t){setTimeout(()=>{t.detail.complete()},3e3)}getExercisesByYear(t){return this.data.getExercisesByYear(t)}}return n.\u0275fac=function(t){return new(t||n)(e.Y36(m.D))},n.\u0275cmp=e.Xpm({type:n,selectors:[["app-home"]],decls:23,vars:4,consts:[[3,"translucent"],[3,"fullscreen"],["slot","fixed",3,"ionRefresh"],["collapse","condense"],["size","large"],[3,"exercise",4,"ngFor","ngForOf"],[3,"exercise"]],template:function(t,r){1&t&&(e.TgZ(0,"ion-header",0)(1,"ion-toolbar")(2,"ion-title"),e._uU(3," Probleme din subiectele de anii trecuti, rezolvate de Chat GPT-3 "),e.qZA()()(),e.TgZ(4,"ion-content",1)(5,"ion-refresher",2),e.NdJ("ionRefresh",function(C){return r.refresh(C)}),e._UZ(6,"ion-refresher-content"),e.qZA(),e.TgZ(7,"ion-header",3)(8,"ion-toolbar")(9,"ion-title",4),e._uU(10," Probleme din subiectele de anii trecuti, rezolvate de Chat GPT-3 "),e.qZA()()(),e.TgZ(11,"ion-list")(12,"ion-list-header")(13,"ion-label")(14,"h2"),e._uU(15,"2017"),e.qZA()()(),e.YNc(16,f,1,1,"app-message",5),e.qZA(),e.TgZ(17,"ion-list")(18,"ion-list-header")(19,"ion-label")(20,"h2"),e._uU(21,"2018"),e.qZA()()(),e.YNc(22,x,1,1,"app-message",5),e.qZA()()),2&t&&(e.Q6J("translucent",!0),e.xp6(4),e.Q6J("fullscreen",!0),e.xp6(12),e.Q6J("ngForOf",r.getExercisesByYear(2017)),e.xp6(6),e.Q6J("ngForOf",r.getExercisesByYear(2018)))},dependencies:[a.sg,i.W2,i.Gu,i.Q$,i.q_,i.yh,i.nJ,i.Wo,i.wd,i.sr,u]}),n})()}];let P=(()=>{class n{}return n.\u0275fac=function(t){return new(t||n)},n.\u0275mod=e.oAB({type:n}),n.\u0275inj=e.cJS({imports:[c.Bz.forChild(h),c.Bz]}),n})(),M=(()=>{class n{}return n.\u0275fac=function(t){return new(t||n)},n.\u0275mod=e.oAB({type:n}),n.\u0275inj=e.cJS({imports:[a.ez,g.u5,i.Pc,c.Bz]}),n})(),_=(()=>{class n{}return n.\u0275fac=function(t){return new(t||n)},n.\u0275mod=e.oAB({type:n}),n.\u0275inj=e.cJS({imports:[a.ez,g.u5,i.Pc,M,P]}),n})()}}]);